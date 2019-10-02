# Author: petter.strandmark@gmail.com (Petter Strandmark)

MESSAGE("-- Adding self train")
ADD_EXECUTABLE(train
                train.cc
                policy_value_model.cc
                surakarta.cc
               ${MCTS_HEADERS})
target_link_libraries(train ${MY_LIBRARY_DEPENDENCIES})

MESSAGE("-- Adding net play: surakarta")
ADD_EXECUTABLE(play_net
                play_net.cc
                policy_value_model.cc
                surakarta.cc
               ${MCTS_HEADERS})
target_link_libraries(play_net ${MY_LIBRARY_DEPENDENCIES})

MESSAGE("-- Adding play: surakarta")
ADD_EXECUTABLE(play
                play.cc
                policy_value_model.cc
                surakarta.cc
               ${MCTS_HEADERS})
target_link_libraries(play ${MY_LIBRARY_DEPENDENCIES})
