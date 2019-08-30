#include "surakarta.h"
#include "mcts.h"
#include <vector>
#include <utility>
#include <iostream>
#include <unordered_map>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <cstdlib>
#include <cassert>

#define PORT  8999
#define BUFFER 2048

SurakartaState::Move fromsocket(int fd) {
    char buffer[BUFFER];
    auto l = recv(fd, buffer, BUFFER, 0);
    if (l == -1) {
        perror(strerror(errno));
        exit(-1);
    }
    assert(l == 7);
    buffer[7] = '\0';
    std::stringstream str;
    str << buffer;
    SurakartaState::Move m = SurakartaState::no_move;
    str >> m;
    return m;
}

void tosocket(int fd, SurakartaState::Move move) {
    std::stringstream m;
    m << move;
    auto str = m.str();
    if (send(fd, str.c_str(), str.length(), 0) == -1) {
        perror(strerror(errno));
        exit(EXIT_FAILURE);
    }
}

void main_program()
{
    auto serverfd = socket(AF_INET, SOCK_STREAM, 0);
    if (serverfd == -1) {
        perror(strerror(errno));
        exit(EXIT_FAILURE);
    }

    decltype(serverfd) fd;
    struct sockaddr_in src_addr, des_addr;

    bool self_play = false;
    bool counter_first = false;
    bool should_move = false;
    bool net_competition = true;


    if (self_play || !counter_first) {
        should_move = true;
    }

    memset(&src_addr, '\0', sizeof(src_addr));

    src_addr.sin_family= AF_INET;
    src_addr.sin_addr.s_addr = INADDR_ANY;
    src_addr.sin_port = htons( PORT );

    socklen_t addrlen = sizeof(src_addr);
    if (net_competition) {
        if (bind(serverfd, (struct sockaddr *)&src_addr, sizeof(src_addr)) < 0)
            perror(strerror(errno));
        if (listen(serverfd, 1) != 0)
            perror(strerror(errno));
        if ((fd = accept(serverfd, (struct sockaddr *)&des_addr, &addrlen)) < 0)
            perror(strerror(errno));
        if (counter_first)
            send(fd, static_cast<const void *>("1"), static_cast<size_t>(1), 0);
    }

    MCTS::ComputeOptions player1_options, player2_options;
    player1_options.max_time =  10.0;
    player1_options.number_of_threads = 8;
    player1_options.verbose = true;

    player2_options.max_time= 30.0;
    player2_options.number_of_threads = 4;
	player2_options.verbose = true;

	SurakartaState state;
    state.player_to_move = 1;
	while (state.has_moves()) {
		cout << endl << "State: " << state << endl;

		SurakartaState::Move move = SurakartaState::no_move;
		if (should_move) {
                move = MCTS::compute_move(state, player1_options);
                state.do_move(move);
                if (net_competition)
                    tosocket(fd, move);
		} else {
				while (true) {
                    if (net_competition) {
                        move = fromsocket(fd);
                    } else {
                        cout << "Input your move: ";
                        std::cin >> move;
                    }
                        try {
                            std::cout << "counter move: " << move;
                            state.do_move(move);
                            break;
                        }
                        catch (std::exception& ) {
                            cout << "Invalid move." << endl;
                        }
				}
        }
        if (!self_play)
            should_move = !should_move;
    }

    std::cout << endl << "Final state: " << state << endl;

	if (state.get_result(2) == 1.0) {
		cout << "Player 1 wins!" << endl;
	}
	else if (state.get_result(1) == 1.0) {
		cout << "Player 2 wins!" << endl;
	}
	else {
		cout << "Nobody wins!" << endl;
	}
    return;
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
