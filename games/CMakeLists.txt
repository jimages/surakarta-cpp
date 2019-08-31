# Author: petter.strandmark@gmail.com (Petter Strandmark)

ADD_EXECUTABLE(train
                train.cc
                policy_value_model.cc
                surakarta.cc
               ${MCTS_HEADERS})
target_link_libraries(train ${MY_LIBRARY_DEPENDENCIES})
MESSAGE("-- Adding game: surakarta")
