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

    // debug only put a node in the queue if the it is empty.
    if(search_->auxengine_queue_.size() > 0) return;

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

  // kickstart with the root node, no need to wait for it to get visits.
  root_node_->SetAuxEngineMove(0xfffe); // mark root as pending.
  auxengine_mutex_.lock();  
  auxengine_queue_.push(root_node_);
  auxengine_cv_.notify_one();
  auxengine_mutex_.unlock();

  Node* n;

    while (!stop_.load(std::memory_order_acquire)) {
      {
	std::unique_lock<std::mutex> lock(auxengine_mutex_);
	// Wait until there's some work to compute.
	auxengine_cv_.wait(lock, [&] { return stop_.load(std::memory_order_acquire) || !auxengine_queue_.empty(); });
	if (stop_.load(std::memory_order_acquire)) break;
	n = auxengine_queue_.front();
	auxengine_queue_.pop();
      } // release lock
      DoAuxEngine(n);
    }
  LOGFILE << "AuxEngineWorker done";
}

void Search::DoAuxEngine(Node* n) {
  // Calculate depth in a safe way. Return early if root cannot be
  // reached from n.
  // Todo test if this lock is unnecessary when solidtree is disabled.
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
  // When we internally use the moves to extend nodes in the search tree, always use move as seen from the white side.
  // Apply the moves in reversed order to get the proper board state from which we can then make moves in legacy format.
  std::vector<lczero::Move> my_moves;
  std::vector<lczero::Move> my_moves_from_the_white_side;  
    
  for (Node* n2 = n; n2 != root_node_; n2 = n2->GetParent()) {
      my_moves.push_back(n2->GetOwnEdge()->GetMove(flip));
      my_moves_from_the_white_side.push_back(n2->GetOwnEdge()->GetMove());
      flip = !flip;
  }

  // Reverse the order
  std::reverse(my_moves.begin(), my_moves.end());
  std::reverse(my_moves_from_the_white_side.begin(), my_moves_from_the_white_side.end());
    
  ChessBoard my_board = played_history_.Last().GetBoard();
  Position my_position = played_history_.Last();

  // Try exporting a fen instead of messing with individual moves,
  // that way there is no need to convert moves from modern to legacy
  // encoding

  // // old code, use legacy moves      
  // for(auto& move: my_moves) {
  //   if (my_board.flipped()) move.Mirror();
  //   move = my_board.GetLegacyMove(move);
  //   my_board.ApplyMove(move);
  //   my_position = Position(my_position, move);
  //   if (my_board.flipped()) move.Mirror();
  //   // not necessary now that we use GetFen().    
  //   s = s + move.as_string() + " "; 
  //   my_board.Mirror();
  // }

  // modern encoding
  for(auto& move: my_moves) {
    if (my_board.flipped()) move.Mirror();
    my_board.ApplyMove(move);
    my_position = Position(my_position, move);
    if (my_board.flipped()) move.Mirror();
    s = s + move.as_string() + " ";  // only for debugging
    my_board.Mirror();
  }

  if (params_.GetAuxEngineVerbosity() >= 1) {
    LOGFILE << "add pv=" << s << " from root position: " << GetFen(played_history_.Last());
  }
  // s = current_uci_ + " " + s;
  LOGFILE << "trying to get a FEN from my_position";
  s = "position fen " + GetFen(my_position);
  LOGFILE << "got a FEN from my_position";  
  LOGFILE << "input to helper: " << s;
  
  // Before starting, test if stop_ is set
  if (stop_.load(std::memory_order_acquire)) {
    if (params_.GetAuxEngineVerbosity() >= 5) {    
      LOGFILE << "DoAuxEngine caught a stop signal 1.";
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
	  LOGFILE << "DoAuxEngine caught a stop signal 2.";	
	}
        // Send stop, stay in loop to get best response, otherwise it
        // will disturb the next iteration.
        auxengine_os_ << "stop" << std::endl;
      }
    }
  }
  if (stopping) {
    // TODO: actually do use the result, if the depth achieved was the
    // requested depth and the line is actually played.

    // Don't use results of a search that was stopped. LOGFILE << "AUX
    // Search was interrupted, the results will not be used";
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

  flip = played_history_.IsBlackToMove() ^ (depth % 2 == 0);

  auto bestmove_packed_int = Move(token, !flip).as_packed_int();
  int pv_length = 1;
  // depth is distance between root and the starting point for the auxengine
  // params_.GetAuxEngineDepth() is the depth of the requested search
  // The actual PV is often times longer, but don't trust the extra plies. 
  // LOGFILE << "capping PV at length: " << depth + params_.GetAuxEngineDepth() << ", sum of depth = " << depth << " and AuxEngineDepth = " << params_.GetAuxEngineDepth();
  // while(iss >> pv >> std::ws && pv_length < depth + params_.GetAuxEngineDepth()) {
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
	// convert to Modern encoding, update the board and the position
	// For the conversion never flip the board. Only flip the board when you need to apply the move!
	Move m_in_modern_encoding = my_board.GetModernMove(m);

	if (my_board.flipped()) m_in_modern_encoding.Mirror();
	// Should the move applied be modern or legacy, or does it not matter?
	m_in_modern_encoding = my_board.GetModernMove(m_in_modern_encoding);
	// my_board.ApplyMove(m_in_modern_encoding); // Todo verify the correctness here, e.g. by printing a FEN.
	my_board.ApplyMove(m); // Todo verify the correctness here, e.g. by printing a FEN.	
	my_position = Position(my_position, m_in_modern_encoding);

	if (my_board.flipped()) m_in_modern_encoding.Mirror();
	my_board.Mirror();

	// my_moves.push_back(m); // Add the PV to the queue
	my_moves_from_the_white_side.push_back(m_in_modern_encoding); // Add the PV to the queue	
        pv_moves.push_back(m_in_modern_encoding.as_packed_int());
        flip = !flip;
	pv_length++;
      }
    }
  }

  if (pv_moves.size() == 0) {
    if (params_.GetAuxEngineVerbosity() >= 1) {
      LOGFILE << "Warning: the helper did not give a PV, will only use bestmove:" << bestmove_packed_int;
    }
    pv_moves.push_back(bestmove_packed_int);

    // This test will fail whenever the first move in the PV is castle, since pv_moves[0] is in modern encoding while
    // bestmove_packed_int is on legacy encoding
    
  // } else if (pv_moves[0] != bestmove_packed_int) {
  //   // TODO: Is it possible for PV to not match bestmove?
  //   LOGFILE << "error: pv doesn't match bestmove:" << pv_moves[0] << " " << "bm" << bestmove_packed_int;
  //   pv_moves.clear();
  //   pv_moves.push_back(bestmove_packed_int);
  //   // stop here.
  //   return;
  }

  
  std::string debug_string;
  for(int i = 0; i < (int) my_moves_from_the_white_side.size(); i++){
    debug_string = debug_string + my_moves_from_the_white_side[i].as_string() + " ";
  }
  if(played_history_.IsBlackToMove()){
    LOGFILE << "debug info: length of PV given to helper engine: " << depth << " position given to helper: " << s << " black to move at root, length of my_moves_from_the_white_side " << my_moves_from_the_white_side.size() << " my_moves_from_the_white_side: " << debug_string;
  } else {
    LOGFILE << "debug info: length of PV given to helper engine: " << depth << " position given to helper: " << s << " white to move at root, length of my_moves_from_the_white_side " << my_moves_from_the_white_side.size() << " my_moves_from_the_white_side: " << debug_string;    
  }
  
  fast_track_extend_and_evaluate_queue_mutex_.lock();
  fast_track_extend_and_evaluate_queue_.push(my_moves_from_the_white_side); // push() since it is a queue.
  fast_track_extend_and_evaluate_queue_mutex_.unlock();
}

void Search::AuxWait() REQUIRES(threads_mutex_) {
  LOGFILE << "AuxWait start";
  while (!auxengine_threads_.empty()) {
    auxengine_threads_.back().join();
    auxengine_threads_.pop_back();
  }
  // Threading/Locking:
  // - Search::Wait is holding threads_mutex_.
  // - SearchWorker threads are guaranteed done by Search::Wait
  // - Above code guarantees auxengine_threads_ are done.
  // - This is the only thread left that can modify auxengine_queue_
  // - Take the lock anyways to be safe.
  std::unique_lock<std::mutex> lock(auxengine_mutex_);
  LOGFILE << "Summaries per move: auxengine_queue_ size at the end of search: " << auxengine_queue_.size()
      << " Average duration " << (auxengine_num_evals ? (auxengine_total_dur / auxengine_num_evals) : -1.0f) << "ms"
      << " Number of evals " << auxengine_num_evals
      << " Number of added nodes " << auxengine_num_updates;
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
  // Empty the other queue.
  fast_track_extend_and_evaluate_queue_mutex_.lock();
  if(fast_track_extend_and_evaluate_queue_.empty()){
    LOGFILE << "No PVs in the fast_track_extend_and_evaluate_queue";
  }
  while (!fast_track_extend_and_evaluate_queue_.empty()){
    LOGFILE << "Removing obsolete PV from queue";
    fast_track_extend_and_evaluate_queue_.pop();
  }
  fast_track_extend_and_evaluate_queue_mutex_.unlock();  
  LOGFILE << "AuxWait done";
}

}  // namespace lczero
