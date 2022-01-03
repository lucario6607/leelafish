/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2022 The LCZero Authors

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

void SearchWorker::AuxMaybeEnqueueNode(Node* n, int source) {
  // the caller (DoBackupUpdate()->DoBackupUpdateSingleNode()) has a lock on search_->nodes_mutex_, so no other thread will change n right now.

  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE
      << "AuxMaybeEnqueueNode() picked node: " << n->DebugString() 
      << " for the persistent_queue_of_nodes which has size: "
      << search_->search_stats_->persistent_queue_of_nodes.size()
      << " The source was " << source;

  n->SetAuxEngineMove(0xfffe); // magic for pending
  search_->search_stats_->persistent_queue_of_nodes.push(n);
  search_->search_stats_->source_of_queued_nodes.push(source);
  search_->auxengine_cv_.notify_one();
  search_->auxengine_mutex_.unlock();
}

void Search::AuxEngineWorker() {
  int number_of_pvs_delivered = 0;

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
    // Set the time to use, based on the UCI parameter
    search_stats_->AuxEngineTime = params_.GetAuxEngineTime();
    search_stats_->AuxEngineThreshold = params_.GetAuxEngineThreshold();
  } else {
    // TODO implement a "newgame" field in search_stats_.
    // If newgame then engine.cc will set search_stats_->AuxEngineTime to zero, and we have to set it to the parameter value (which is not available to engine.cc)
    if(search_stats_->AuxEngineTime == 0) {
      search_stats_->AuxEngineTime = params_.GetAuxEngineTime();
      if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE
	  << "Setting AuxEngineTime to "
	  << search_stats_->AuxEngineTime << " since it was zero";
    }
    
    // purge obsolete nodes in the queue, if any. The even elements are the actual nodes, the odd elements is root if the preceding even element is still a relevant node.
    if(search_stats_->persistent_queue_of_nodes.size() > 0){
      int number_of_nodes_before_purging = int(search_stats_->persistent_queue_of_nodes.size() / 2);
      std::queue<Node*> persistent_queue_of_nodes_temp_;
      long unsigned int my_size = search_stats_->persistent_queue_of_nodes.size();      
      for(long unsigned int i=0; i < my_size; i = i + 2){
	Node * n = search_stats_->persistent_queue_of_nodes.front();
	search_stats_->persistent_queue_of_nodes.pop();
	Node * n_parent = search_stats_->persistent_queue_of_nodes.front();
	search_stats_->persistent_queue_of_nodes.pop();
	if(n_parent == root_node_){
	  // node is still relevant
	  persistent_queue_of_nodes_temp_.push(n);
	}
      }
      // update search_stats_->persistent_queue_of_nodes
      my_size = persistent_queue_of_nodes_temp_.size();
      for(long unsigned int i=0; i < my_size; i++){      
    	search_stats_->persistent_queue_of_nodes.push(persistent_queue_of_nodes_temp_.front());
    	persistent_queue_of_nodes_temp_.pop();
      }
      if (params_.GetAuxEngineVerbosity() >= 5)
	LOGFILE << "Purged " << number_of_nodes_before_purging - search_stats_->persistent_queue_of_nodes.size()
		<< " nodes from the query queue due to the move selected by the opponent. " << search_stats_->persistent_queue_of_nodes.size()
		<< " nodes remain in the queue.";
    }

    // Also purge stale nodes from the _added_ queue.
    if(search_stats_->nodes_added_by_the_helper.size() > 0){
      int number_of_nodes_before_purging = int(search_stats_->nodes_added_by_the_helper.size() / 2);
      std::queue<Node*> nodes_added_by_the_helper_temp_;
      long unsigned int my_size = search_stats_->nodes_added_by_the_helper.size();      
      for(long unsigned int i=0; i < my_size; i = i + 2){
	Node * n = search_stats_->nodes_added_by_the_helper.front();
	search_stats_->nodes_added_by_the_helper.pop();
	Node * n_parent = search_stats_->nodes_added_by_the_helper.front();
	search_stats_->nodes_added_by_the_helper.pop();
	if(n_parent == root_node_){
	  // node is still relevant
	  nodes_added_by_the_helper_temp_.push(n);
	}
      }
      // update search_stats_->nodes_added_by_the_helper
      my_size = nodes_added_by_the_helper_temp_.size();
      for(long unsigned int i=0; i < my_size; i++){      
    	search_stats_->nodes_added_by_the_helper.push(nodes_added_by_the_helper_temp_.front());
    	nodes_added_by_the_helper_temp_.pop();
      }
      if (params_.GetAuxEngineVerbosity() >= 5)
	LOGFILE << "Purged " << number_of_nodes_before_purging - search_stats_->nodes_added_by_the_helper.size()
		<< " stale nodes from the queue of nodes added by the auxillary helper due to the move seleted by the opponent. " << search_stats_->nodes_added_by_the_helper.size()
		<< " nodes remain in the queue of nodes added by the auxillary helper.";
    }
  }

  // Kickstart with the root node, no need to wait for it to get some
  // amount of visits. Except if root is not yet expanded, or lacks
  // edges for any other reason (e.g. being terminal), in which case we
  // should wait.
  nodes_mutex_.lock(); // write lock
  if(root_node_->GetNumEdges() > 0){
    // root is extended.
    root_node_->SetAuxEngineMove(0xfffe); // mark root as pending and queue it
    auxengine_mutex_.lock(); 
    search_stats_->persistent_queue_of_nodes.push(root_node_);
    search_stats_->source_of_queued_nodes.push(3);
    auxengine_cv_.notify_one();
    auxengine_mutex_.unlock();
    // if root has children, queue those too. (since we can only queue _nodes_ we can not unconditionally queue all _edges_ of root).
    
  }
  nodes_mutex_.unlock(); // write unlock

  Node* n;
  while (!stop_.load(std::memory_order_acquire)) {
    {
  	std::unique_lock<std::mutex> lock(auxengine_mutex_);
  	// Wait until there's some work to compute.
  	auxengine_cv_.wait(lock, [&] { return stop_.load(std::memory_order_acquire) || !search_stats_->persistent_queue_of_nodes.empty(); });
  	if (stop_.load(std::memory_order_acquire)) {
  	  auxengine_mutex_.unlock(); 	
  	  break;
  	}
  	n = search_stats_->persistent_queue_of_nodes.front();
  	search_stats_->persistent_queue_of_nodes.pop();
    } // release lock
    DoAuxEngine(n);
    ++number_of_pvs_delivered;
  }
  auxengine_mutex_.unlock();
  if (params_.GetAuxEngineVerbosity() >= 1) LOGFILE  
    << "AuxEngineWorker done, delivered " << number_of_pvs_delivered << " PVs.";
}

void Search::DoAuxEngine(Node* n) {
  // before trying to take a lock on nodes_mutex_, always check if search has stopped, in which we return early
  if(stop_.load(std::memory_order_acquire)) {
  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "DoAuxEngine caught a stop signal beforing doing anything.";
    return;
  }
  nodes_mutex_.lock_shared();
  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "DoAuxEngine() called for node" << n->DebugString();
  nodes_mutex_.unlock_shared();  
  // Calculate depth in a safe way. Return early if root cannot be
  // reached from n.
  // Todo test if this lock is unnecessary when solidtree is disabled.
  int depth = 0;
  if(n != root_node_){
    if(stop_.load(std::memory_order_acquire)) {
      if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "DoAuxEngine caught a stop signal while calculating depth.";
      return;
    }
    nodes_mutex_.lock_shared();
    for (Node* n2 = n; n2 != root_node_; n2 = n2->GetParent()) {
      depth++;
    }
    nodes_mutex_.unlock_shared();    
  }

  std::string s = "";
  bool flip = played_history_.IsBlackToMove() ^ (depth % 2 == 0);

  // To get the moves in UCI format, we have to construct a board, starting from root and then apply the moves.
  // Traverse up to root, and store the moves in a vector.
  // When we internally use the moves to extend nodes in the search tree, always use move as seen from the white side.
  // Apply the moves in reversed order to get the proper board state from which we can then make moves in legacy format.
  std::vector<lczero::Move> my_moves;
  std::vector<lczero::Move> my_moves_from_the_white_side;  

  if(n != root_node_){
    if(stop_.load(std::memory_order_acquire)) {
      if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "DoAuxEngine caught a stop signal while populating my_moves.";
      return;
    }
    nodes_mutex_.lock_shared();  
    for (Node* n2 = n; n2 != root_node_; n2 = n2->GetParent()) {
      my_moves.push_back(n2->GetOwnEdge()->GetMove(flip));
      my_moves_from_the_white_side.push_back(n2->GetOwnEdge()->GetMove());
      flip = !flip;
    }
    nodes_mutex_.unlock_shared();
  }

  // Reverse the order
  std::reverse(my_moves.begin(), my_moves.end());
  std::reverse(my_moves_from_the_white_side.begin(), my_moves_from_the_white_side.end());
    
  ChessBoard my_board = played_history_.Last().GetBoard();
  Position my_position = played_history_.Last();

  // modern encoding
  for(auto& move: my_moves) {
    if (my_board.flipped()) move.Mirror();
    my_board.ApplyMove(move);
    my_position = Position(my_position, move);
    if (my_board.flipped()) move.Mirror();
    s = s + move.as_string() + " ";  // only for debugging
    my_board.Mirror();
  }

  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "add pv=" << s << " from root position: " << GetFen(played_history_.Last());
  s = "position fen " + GetFen(my_position);
  
  // Before starting, test if stop_ is set
  if (stop_.load(std::memory_order_acquire)) {
    if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "DoAuxEngine caught a stop signal 1.";
    return;
  }

  auto auxengine_start_time = std::chrono::steady_clock::now();
  auxengine_os_ << s << std::endl;
  // use go depth when there are few pieces left on the board. 20 pieces hardcoded for now.
  if((my_board.ours() | my_board.theirs()).count() < 20){
    auxengine_os_ << "go depth " << params_.GetAuxEngineDepth() << std::endl;
  } else {
    auxengine_os_ << "go movetime " << search_stats_->AuxEngineTime << std::endl;
  }

  auxengine_stopped_mutex_.lock();
  if(auxengine_stopped_){
    auxengine_stopped_ = false;    
  }
  auxengine_stopped_mutex_.unlock();

  std::string prev_line;
  std::string line;
  std::string token;
  bool stopping = false;
  while(std::getline(auxengine_is_, line)) {
    if (params_.GetAuxEngineVerbosity() >= 9) {
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
	if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "DoAuxEngine caught a stop signal 2.";	
        // (unless someone else already has sent stop) send stop,
	// stay in loop to get best response, otherwise it
        // will disturb the next iteration.
	// only send stop if we are the first to detect that search has stopped.
	auxengine_stopped_mutex_.lock();
	if(!auxengine_stopped_){
	  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "DoAuxEngine() Stopping the A/B helper Start";
	  auxengine_os_ << "stop" << std::endl; // stop the A/B helper
	  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "DoAuxEngine() Stopping the A/B helper Stop";
	  auxengine_stopped_ = true;
	}
	auxengine_stopped_mutex_.unlock();	
      }
    }
  }
  if (stopping) {
    // TODO: actually do use the result, if the depth achieved was the
    // requested depth and the line is actually played.

    // For now don't use results of a search that was stopped.
    if (params_.GetAuxEngineVerbosity() >= 2) LOGFILE << "AUX Search was interrupted, the results will not be used";
    return;
  }
  auxengine_stopped_mutex_.lock();
  auxengine_stopped_ = true; // stopped means "not running". It does not mean it was stopped prematurely.
  auxengine_stopped_mutex_.unlock();
  
  if (params_.GetAuxEngineVerbosity() >= 9) {
    LOGFILE << "pv:" << prev_line;
    LOGFILE << "bestanswer:" << token;
  }
  if(prev_line == ""){
    if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Empty PV, returning early from doAuxEngine().";
    return;
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
  // depth is distance between root and the starting point for the
  // auxengine.
  // depth_reached records the depth the helper claim to have search.
  // The PV is capped at this length (and can be shortened again in PreExt..()

  fast_track_extend_and_evaluate_queue_mutex_.lock(); // lock this queue before starting to modify it

  int pv_length = 1;
  int depth_reached = 0;

  while(iss >> pv >> std::ws) {
    if (pv == "depth") {
      // Figure out which depth was reached (can be zero).
      iss >> depth_reached >> std::ws;
      if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Reached depth: " << depth_reached << " for node with depth: " << depth;
    }
    if (pv == "pv") {
      while(iss >> pv >> std::ws &&
	    pv_length < depth_reached &&
	    pv_length < params_.GetAuxEngineFollowPvDepth()) {
        Move m;
        if (!Move::ParseMove(&m, pv, !flip)) {	
          if (params_.GetAuxEngineVerbosity() >= 1) LOGFILE << "Ignoring bad pv move: " << pv;
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
	// if(pv_length == max_pv_length) break;
	pv_length++;
      }
    }
  }

  if (pv_moves.size() == 0) {
    if (params_.GetAuxEngineVerbosity() >= 1) LOGFILE << "Warning: the helper did not give a PV, will only use bestmove:" << bestmove_packed_int;
    pv_moves.push_back(bestmove_packed_int);
  }

  if (params_.GetAuxEngineVerbosity() >= 9){
    std::string debug_string;
    for(int i = 0; i < (int) my_moves_from_the_white_side.size(); i++){
      debug_string = debug_string + my_moves_from_the_white_side[i].as_string() + " ";
    }
    if(played_history_.IsBlackToMove()){
      LOGFILE << "debug info: length of PV given to helper engine: " << depth << " position given to helper: " << s << " black to move at root, length of my_moves_from_the_white_side " << my_moves_from_the_white_side.size() << " my_moves_from_the_white_side: " << debug_string;
    } else {
      LOGFILE << "debug info: length of PV given to helper engine: " << depth << " position given to helper: " << s << " white to move at root, length of my_moves_from_the_white_side " << my_moves_from_the_white_side.size() << " my_moves_from_the_white_side: " << debug_string;    
    }
  }
  
  fast_track_extend_and_evaluate_queue_.push(my_moves_from_the_white_side); // I think push() means push_back for queues.
  fast_track_extend_and_evaluate_queue_mutex_.unlock();

  if (params_.GetAuxEngineVerbosity() >= 9) LOGFILE << "Returning from DoAuxEngine()";
}

void Search::AuxWait() {  
  if (params_.GetAuxEngineVerbosity() >= 7) LOGFILE << "AuxWait start for thread: " << std::hash<std::thread::id>{}(std::this_thread::get_id());

  while (!auxengine_threads_.empty()) {
    Mutex::Lock lock(threads_mutex_);
    if (params_.GetAuxEngineVerbosity() >= 7) LOGFILE << "AuxWait about to pop threads";
    auxengine_threads_.back().join();
    auxengine_threads_.pop_back();
  }
  if(search_stats_->persistent_queue_of_nodes.size() > 100){
    // decrease the number of queued nodes that comes the via the threshold, by increasing the threshold
    search_stats_->AuxEngineThreshold = search_stats_->AuxEngineThreshold * 1.1;
  }
  if(search_stats_->persistent_queue_of_nodes.size() == 0){
    // increase the number of queued nodes that comes the via the threshold, by decreasing the threshold. Don't decrease it under the parameter value.
    search_stats_->AuxEngineThreshold = std::min(params_.GetAuxEngineThreshold(), int(search_stats_->AuxEngineThreshold * 0.9));
  }
  ChessBoard my_board = played_history_.Last().GetBoard();
  if((my_board.ours() | my_board.theirs()).count() >= 20){
    if(search_stats_->persistent_queue_of_nodes.size() == 0){     
    // if(search_stats_->persistent_queue_of_nodes.size() < auxengine_num_evals * 0.1){ 
      // increase time if more than 90% of all queued nodes were delivered
      search_stats_->AuxEngineTime = search_stats_->AuxEngineTime * 1.1;
    }
    if(search_stats_->persistent_queue_of_nodes.size() > 100 && search_stats_->AuxEngineTime > 60){
      // Don't go too low for a query.
      
    // if(search_stats_->persistent_queue_of_nodes.size() > auxengine_num_evals * 0.5){
      // decrease time if queue is greater than half of the number of delivered PVs
      search_stats_->AuxEngineTime = search_stats_->AuxEngineTime * 0.9;
    }
    // Time based queries    
    LOGFILE << "Summaries per move: (Time based queries) persistent_queue_of_nodes size at the end of search: " << search_stats_->persistent_queue_of_nodes.size()
      << " Average duration " << (auxengine_num_evals ? (auxengine_total_dur / auxengine_num_evals) : -1.0f) << "ms"
      << " New AuxEngineTime for next iteration " << search_stats_->AuxEngineTime
      << " New AuxEngineThreshold for next iteration " << search_stats_->AuxEngineThreshold
      << " Number of evals " << auxengine_num_evals
      << " Number of added nodes " << auxengine_num_updates;
  } else {
    // Depth based queries
    LOGFILE << "Summaries per move: (Depth=" << params_.GetAuxEngineFollowPvDepth() << ") persistent_queue_of_nodes size at the end of search: " << search_stats_->persistent_queue_of_nodes.size()
      << " Average duration " << (auxengine_num_evals ? (auxengine_total_dur / auxengine_num_evals) : -1.0f) << "ms"
      << " New AuxEngineThreshold for next iteration " << search_stats_->AuxEngineThreshold      
      << " Number of evals " << auxengine_num_evals
      << " Number of added nodes " << auxengine_num_updates;
  }
  
  // Empty the other queue.
  fast_track_extend_and_evaluate_queue_mutex_.lock();
  if(fast_track_extend_and_evaluate_queue_.empty()){
    if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "No PVs in the fast_track_extend_and_evaluate_queue";
  } else {
    if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << fast_track_extend_and_evaluate_queue_.size() << " possibly obsolete PV:s in the queue.";
    fast_track_extend_and_evaluate_queue_ = {}; // should be faster than pop() but it is safe?
    while (!fast_track_extend_and_evaluate_queue_.empty()) {
      fast_track_extend_and_evaluate_queue_.pop();
    }
    if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "Number of PV:s in the queue=" << fast_track_extend_and_evaluate_queue_.size();
  }
  fast_track_extend_and_evaluate_queue_mutex_.unlock();  
  if (params_.GetAuxEngineVerbosity() >= 5) LOGFILE << "AuxWait done";
}

}  // namespace lczero
