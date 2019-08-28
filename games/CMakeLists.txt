# Author: petter.strandmark@gmail.com (Petter Strandmark)

ADD_EXECUTABLE(surakarta
                surakarta.cc
                policy_value_model.cc
               ${MCTS_HEADERS})
target_link_libraries(surakarta ${MY_LIBRARY_DEPENDENCIES})
MESSAGE("-- Adding game: surakarta")
