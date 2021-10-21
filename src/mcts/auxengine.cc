/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "mcts/search.h"

#include <algorithm>
#include <boost/process.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>

#include "mcts/node.h"
#include "neural/cache.h"
#include "neural/encoder.h"
#include "utils/fastmath.h"
#include "utils/random.h"

namespace lczero {

boost::process::ipstream Search::auxengine_is_;
boost::process::opstream Search::auxengine_os_;
boost::process::child Search::auxengine_c_;
bool Search::auxengine_ready_ = false;

void Search::OpenAuxEngine() REQUIRES(threads_mutex_) {
  if (params_.GetAuxEngineFile() == "") return;
  auxengine_threads_.emplace_back([this]() { AuxEngineWorker(); });
}

void SearchWorker::AuxMaybeEnqueueNode(Node* n) {
  if (params_.GetAuxEngineFile() != "" &&
      n->GetN() >= (uint32_t) params_.GetAuxEngineThreshold() &&
      n->GetAuxEngineMove() == 0xffff &&
      !n->IsTerminal() &&
      n->HasChildren()) {
    n->SetAuxEngineMove(0xfffe); // TODO: magic for pending
    std::lock_guard<std::mutex> lock(search_->auxengine_mutex_);
    search_->auxengine_queue_.push(n);
    search_->auxengine_cv_.notify_one();
  }
}

void Search::AuxEngineWorker() {
  if (!auxengine_ready_) {
    auxengine_c_ = boost::process::child(params_.GetAuxEngineFile(), boost::process::std_in < auxengine_os_, boost::process::std_out > auxengine_is_);
    {
      std::istringstream iss(params_.GetAuxEngineOptions());
      std::string token;
      while(std::getline(iss, token, '=')) {
        std::ostringstream oss;
        oss << "setoption name " << token;
        std::getline(iss, token, ';');
        oss << " value " << token;
        LOGFILE << oss.str();
        auxengine_os_ << oss.str() << std::endl;
      }
      auxengine_os_ << "uci" << std::endl;
    }
    std::string line;
    while(std::getline(auxengine_is_, line)) {
      LOGFILE << line;
      std::istringstream iss(line);
      std::string token;
      iss >> token >> std::ws;
      if (token == "uciok") {
        break;
      } else if (token == "option") {
        iss >> token >> std::ws;
        if (token == "name") {
          iss >> token >> std::ws;
          if (token == "SyzygyPath" && syzygy_tb_) {
            std::ostringstream oss;
            oss << "setoption name SyzygyPath value " << syzygy_tb_->get_paths();
            LOGFILE << oss.str();
            auxengine_os_ << oss.str() << std::endl;
          }
        }
      }
    }
    auxengine_ready_ = true;
  }
  if (current_position_fen_ == "") {
    current_position_fen_ = ChessBoard::kStartposFen; // TODO [HE: what is there todo?]
  }
  if (current_position_moves_.size()) {
    for (auto i = current_position_moves_.size(); i-- > 0;) {
      current_uci_ = current_position_moves_[i] + " " + current_uci_;
    }
  }
  current_uci_ = "position fen " + current_position_fen_ + " moves " + current_uci_;
  LOGFILE << current_uci_;

  Node* n;

  // TODO handle this: 1019 15:31:45.540938 140297824630528 ../../src/mcts/stoppers/stoppers.cc:195] Only one possible move. Moving immediately. DONE with if(root_node_->GetNumEdges() > 1){
  // TODO handle this: 1019 16:53:49.746308 139657706731264 ../../src/mcts/stoppers/stoppers.cc:199] At most one non losing move, stopping search.  
  // if(root_node_->GetNumEdges() > 1){
    while (!stop_.load(std::memory_order_acquire)) {
      {
	std::unique_lock<std::mutex> lock(auxengine_mutex_);
	// Wait until there's some work to compute.
	auxengine_cv_.wait(lock, [&] { return stop_.load(std::memory_order_acquire) || !auxengine_queue_.empty(); });
	if (stop_.load(std::memory_order_acquire)) break;
	n = auxengine_queue_.front();
	auxengine_queue_.pop();
      } // release lock
      /* LOGFILE << "AuxEngineWorker: DoAuxEngine() called"; */
      DoAuxEngine(n);
      /* LOGFILE << "AuxEngineWorker: DoAuxEngine() returned"; */
    }
  LOGFILE << "AuxEngineWorker done";
}

void Search::DoAuxEngine(Node* n) {
  if (n == nullptr){
    LOGFILE << "at DoAuxEngine: called with null pointer.";
    return;
  }
  
  if (n->GetAuxEngineMove() < 0xfffe) {
    LOGFILE << "at DoAuxEngine: called with magic node.";
    return;
  }

  // Calculate depth in a safe way. Return early if root cannot be
  // reached from n.
  nodes_mutex_.lock();
  int depth = 0;
  for (Node* n2 = n; n2 != root_node_; n2 = n2->GetParent()) {
    depth++;
    if(n2 == nullptr){
      LOGFILE << "1. Could not reach root";
      nodes_mutex_.unlock();
      return;
    } 
  }
  nodes_mutex_.unlock();

  std::string s = "";
  bool flip = played_history_.IsBlackToMove() ^ (depth % 2 == 0);

  // To get the moves in UCI format, we have to construct a board, starting from root and then apply the moves.
  // Traverse up to root, and store the moves in a vector.
  // Apply the moves in reversed order to get the proper board state from which we can then make moves in legacy format.
  std::vector<lczero::Move> my_moves;
    
  for (Node* n2 = n; n2 != root_node_; n2 = n2->GetParent()) {
      my_moves.push_back(n2->GetOwnEdge()->GetMove(flip));
      flip = !flip;
  }

  // Reverse the order
  std::reverse(my_moves.begin(), my_moves.end());
    
  ChessBoard my_board = played_history_.Last().GetBoard();

  for(auto& move: my_moves) {
    if (my_board.flipped()) move.Mirror();
    move = my_board.GetLegacyMove(move);
    my_board.ApplyMove(move);
    if (my_board.flipped()) move.Mirror();
    /* LOGFILE << "Move as UCI: " << move.as_string(); */
    s = s + move.as_string() + " ";
    my_board.Mirror();
  }
    
  if (params_.GetAuxEngineVerbosity() >= 1) {
    LOGFILE << "add pv=" << s;
  }
  s = current_uci_ + " " + s;

  // Before starting, test if stop_ is set
  if (stop_.load(std::memory_order_acquire)) {
    if (params_.GetAuxEngineVerbosity() >= 5) {    
      LOGFILE << "DoAuxEngine caught a stop signal";
    }
    return;
  }

  auto auxengine_start_time = std::chrono::steady_clock::now();
  auxengine_os_ << s << std::endl;
  auxengine_os_ << "go depth " << params_.GetAuxEngineDepth() << std::endl;
  std::string prev_line;
  std::string line;
  std::string token;
  bool stopping = false;
  // TODO: while waiting for getline() we do not listen for the stop
  // signal, so shutting down search will be delayed by at most the
  // time between the info lines of the A/B helper. Since we will not
  // use that result anyway, it would be cleaner to return fast and
  // stop the A/B helper when we have time to do so.
  while(std::getline(auxengine_is_, line)) {
    if (params_.GetAuxEngineVerbosity() >= 2) {
      LOGFILE << "auxe:" << line;
    }
    std::istringstream iss(line);
    iss >> token >> std::ws;
    if (token == "bestmove") {
      iss >> token;
      break;
    }
    prev_line = line;

    // Don't send a second stop command
    if (!stopping) {
      stopping = stop_.load(std::memory_order_acquire);
      if (stopping) {
	if (params_.GetAuxEngineVerbosity() >= 5) {
	  LOGFILE << "DoAuxEngine caught a stop signal";	
	}
        // Send stop, stay in loop to get best response, otherwise it
        // will disturb the next iteration.
        auxengine_os_ << "stop" << std::endl;
      }
    }
  }
  if (stopping) {
    // Don't use results of a search that was stopped.
    /* LOGFILE << "AUX Search was interrupted, the results will not be used"; */
    return;
  }
  if (params_.GetAuxEngineVerbosity() >= 1) {
    LOGFILE << "pv:" << prev_line;
    LOGFILE << "bestanswer:" << token;
  }
  if (!auxengine_c_.running()) {
    LOGFILE << "AuxEngine died!";
    throw Exception("AuxEngine died!");
  }
  auto auxengine_dur =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - auxengine_start_time)
      .count();
  auxengine_total_dur += auxengine_dur;
  auxengine_num_evals++;
  std::istringstream iss(prev_line);
  std::string pv;
  std::vector<uint16_t> pv_moves;

  // TODO only store the moves from the white side here, since it
  // appears the value in edge.GetMove() is from the white side.
  
  // reset flip needed? Not sure but should not hurt.
  flip = played_history_.IsBlackToMove() ^ (depth % 2 == 0);
  /* // For some reason the flip is wrong if PV is empty and black is to move (that is if n = root_node_) */
  /* if(n == root_node_ && played_history_.IsBlackToMove()){ */
  /*   flip = !flip; */
  /* } */
  // flip has the current state up to the point where teh PV from aux engine starts, so no reason to set it again.
  
  auto bestmove_packed_int = Move(token, !flip).as_packed_int();
  while(iss >> pv >> std::ws) {
    if (pv == "pv") {
      while(iss >> pv >> std::ws) {
        Move m;
        if (!Move::ParseMove(&m, pv, !flip)) {	
          if (params_.GetAuxEngineVerbosity() >= 2) {
            LOGFILE << "Ignore bad pv move: " << pv;
          }
          break;
        }
        pv_moves.push_back(m.as_packed_int());
        flip = !flip;
      }
    }
  }

  if (pv_moves.size() == 0) {
    if (params_.GetAuxEngineVerbosity() >= 1) {
      LOGFILE << "Warning: the helper did not give a PV, will only use bestmove:" << bestmove_packed_int;
    }
    pv_moves.push_back(bestmove_packed_int);
  } else if (pv_moves[0] != bestmove_packed_int) {
    // TODO: Is it possible for PV to not match bestmove?
    LOGFILE << "error: pv doesn't match bestmove:" << pv_moves[0] << " " << "bm" << bestmove_packed_int;
    pv_moves.clear();
    pv_moves.push_back(bestmove_packed_int);
  }
  // Take the lock and update the P value of the bestmove
  SharedMutex::Lock lock(nodes_mutex_);
  // LOGFILE << "DoAuxEngine: About to call AuxUpdateP()";
  AuxUpdateP(n, pv_moves, 0, my_board);
  // LOGFILE << "DoAuxEngine: AuxUpdateP() finished.";  
}

void Search::AuxUpdateP(Node* n, std::vector<uint16_t> pv_moves, int ply, ChessBoard my_board) {
  // my_board is the position where the node n is.

  if (n->GetAuxEngineMove() < 0xfffe) {
    LOGFILE << "Returning early from AuxUpdateP()";
    // This can happen because nodes are placed in the queue from
    // deepest node first during DoBackupSingeNode
    //if (n->GetAuxEngineMove() != pv_moves[ply]) {
    //  LOGFILE << "already done: curr:" << n->GetAuxEngineMove() << " new:" << pv_moves[ply] << " (error? mismatch)";
    //} else {
    //  LOGFILE << "already done";
    //}
    return;
  }
  
  /* LOGFILE << "At AuxUpdateP() with node:" << n->DebugString(); */
  /* if(my_board.flipped()){ */
  /*   LOGFILE << "my_board is flipped"; */
  /* } else { */
  /*   LOGFILE << "my_board is not flipped"; */
  /* } */
  // unwrap the full set of moves
  std::string s = "";

  // get depth
  int depth = 0;
  Node* n3 = n;

  // This appears to be safe.
  if(n3 == root_node_){
    /* LOGFILE << "at AuxUpdateP: called with root node"; */
  } else {
    while(n3 != root_node_ && n3 != nullptr){
      n3 = n3->GetParent();
      depth++;
    }
    if(n3 == nullptr){
      if (params_.GetAuxEngineVerbosity() >= 2) {
	LOGFILE << "at AuxUpdateP: not able to reach root: old node?";
      }
      return;
    }
  }

  // flip and s are only used to get debugging info.
  // Debugging START
  if(params_.GetAuxEngineVerbosity() >= 2) {
    bool flip = my_board.flipped();  
    for (Node* n2 = n; n2 != root_node_; n2 = n2->GetParent()) {
      s = n2->GetOwnEdge()->GetMove(!flip).as_string() + " " + s;
      flip = !flip;
    }

    if(n == root_node_){
      LOGFILE << "AuxUpdateP() called with ply=" << ply << " to update policy at root and the suggested move here is: " << pv_moves[ply];
    } else {
      LOGFILE << "AuxUpdateP() called with ply=" << ply << " to update policy at a node which can be reached from root via this path: " << s << "and the suggested move (in packed int form) here is: " << pv_moves[ply];
    }
  } // Debugging STOP
    
  for (const auto& edge : n->Edges()) {
    Move move = edge.GetMove();
    /* LOGFILE << "move before converting to legacy castling format: " << move.as_string() << ", " << move.as_packed_int(); */
    move = my_board.GetLegacyMove(move);
    /* LOGFILE << "move after converting to legacy castling format: " << move.as_string() << ", " << move.as_packed_int(); */
    // Delay the application of the move until the right edge is found.
    /* LOGFILE << "move as packed int after converting to uci and back: " << move.as_packed_int(); */

    // Sometimes GetLegacyMove() returns the modern move anyway
    if(move.as_packed_int() == pv_moves[ply] ||
       (move.as_packed_int() == 263 && pv_moves[ply] == 262)
       ) {
      if(move.as_packed_int() == 263 && pv_moves[ply] == 262){
	LOGFILE << "GetLegacyMove() appear to have failed, falling back to a manual hack.";
      }
      auto new_p = edge.GetP() + params_.GetAuxEngineBoost()/100.0f;
      LOGFILE << "Changing P from " << edge.GetP() << " to " << std::min(new_p, 1.0f);
      edge.edge()->SetP(std::min(new_p, 1.0f));
      auxengine_num_updates++;
      if (ply+1 < params_.GetAuxEngineFollowPvDepth() &&
          (uint32_t) ply+1 < pv_moves.size() &&
          edge.HasNode() &&
          !edge.IsTerminal() &&
          edge.node()->HasChildren()) {

	// update the board, now that we have found the correct edge
	if (my_board.flipped()) move.Mirror();
	my_board.ApplyMove(move);
	my_board.Mirror();
	
        AuxUpdateP(edge.node(), pv_moves, ply+1, my_board);
      }
      n->SetAuxEngineMove(pv_moves[ply]);
      return;
    }
  }

  // Leela might have made the node terminal due to repetition, but the AUX engine might not. Only die if there actually are edges.
  if(n->HasChildren()){
    throw Exception("AuxUpdateP: Move not found");
  }
}

void Search::AuxWait() REQUIRES(threads_mutex_) {
  LOGFILE << "AuxWait start";
  while (!auxengine_threads_.empty()) {
    LOGFILE << "Wait for auxengine_threads";
    auxengine_threads_.back().join();
    auxengine_threads_.pop_back();
  }
  LOGFILE << "Done waiting for auxengine_threads to shut down";
  // Threading/Locking:
  // - Search::Wait is holding threads_mutex_.
  // - SearchWorker threads are guaranteed done by Search::Wait
  // - Above code guarantees auxengine_threads_ are done.
  // - This is the only thread left that can modify auxengine_queue_
  // - Take the lock anyways to be safe.
  std::unique_lock<std::mutex> lock(auxengine_mutex_);
  LOGFILE << "AuxWait got a lock. auxengine_queue_ size " << auxengine_queue_.size()
      << " Average duration " << (auxengine_num_evals ? (auxengine_total_dur / auxengine_num_evals) : -1.0f) << "ms"
      << " Number of evals " << auxengine_num_evals
      << " Number of updates " << auxengine_num_updates;
  // TODO: For now with this simple queue method,
  // mark unfinished nodes not done again, and delete the queue.
  // Next search iteration will fill it again.
  while (!auxengine_queue_.empty()) {
    auto n = auxengine_queue_.front();
    assert(n->GetAuxEngineMove() != 0xffff);
    if (n->GetAuxEngineMove() == 0xfffe) {
      n->SetAuxEngineMove(0xffff);
    }
    auxengine_queue_.pop();
  }
  LOGFILE << "AuxWait done";
}

}  // namespace lczero
