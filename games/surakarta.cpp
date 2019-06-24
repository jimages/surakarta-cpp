#include "surakarta.h"
#include <mcts.h>
#include <vector>
#include <utility>
#include <iostream>
const char SurakartaState::player_markers[] = {'*', 'R', 'B'};
const SurakartaState::ChessType SurakartaState::player_chess[] = {SurakartaState::ChessType::Null, SurakartaState::ChessType::Red, SurakartaState::ChessType::Black};
const std::vector<std::pair<int, int>> SurakartaState::outer_loop = {{1,0}, {0,1}, {1,1},
    {2,1}, {3,1}, {4,1}, {5,1}, {4,0}, {4,1}, {4,2}, {4,3}, {4,4}, {4,5},
    {5,4}, {4,4}, {3,4}, {2,4}, {1,4}, {0,4}, {1,5}, {1,4}, {1,3}, {1,2},
    {1,1}};
const std::vector<std::pair<int, int>> SurakartaState::inner_loop = {{2, 0}, {0, 2}, {1, 2},
    {2, 2}, {3, 2}, {4, 2}, {5, 2}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5},
    {5, 3}, {4, 3}, {3, 3}, {2, 3}, {1, 3}, {0, 3}, {2, 5}, {2, 4}, {2, 3}, {2, 2},
   {2, 1}};
const SurakartaState::Move SurakartaState::no_move = {0,{0,0},{0,0}};
using namespace std;
void main_program()
{
	using namespace std;

	bool human_player = true;

	MCTS::ComputeOptions player1_options, player2_options;
	player1_options.max_iterations =  1e5;
    player1_options.number_of_threads = 1;
	player1_options.verbose = true;
    player2_options.max_iterations = 1e5;
    player2_options.number_of_threads = 1;
	player2_options.verbose = true;

	SurakartaState state;
    state.player_to_move = 1;
	while (state.has_moves()) {
		cout << endl << "State: " << state << endl;

		SurakartaState::Move move = SurakartaState::no_move;
		if (state.player_to_move == 1) {
                move = MCTS::compute_move(state, player1_options);
                state.do_move(move);
		}
		else {
			if (human_player) {
				while (true) {
					cout << "Input your move: ";
					move = SurakartaState::no_move;
					cin >> move;
					try {
						state.do_move(move);
						break;
					}
					catch (std::exception& ) {
						cout << "Invalid move." << endl;
					}
				}
			}
			else {
				move = MCTS::compute_move(state, player2_options);
				state.do_move(move);
			}
		}
	}

	cout << endl << "Final state: " << state << endl;

	if (state.get_result(2) == 1.0) {
		cout << "Player 1 wins!" << endl;
	}
	else if (state.get_result(1) == 1.0) {
		cout << "Player 2 wins!" << endl;
	}
	else {
		cout << "Nobody wins!" << endl;
	}
}

int main()
{
	try {
		main_program();
	}
	catch (std::runtime_error& error) {
		std::cerr << "ERROR: " << error.what() << std::endl;
		return 1;
	}
}
