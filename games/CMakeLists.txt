# Author: petter.strandmark@gmail.com (Petter Strandmark)

MACRO (CREATE_EXAMPLE NAME)
	ADD_EXECUTABLE(${NAME}
	               ${NAME}.cpp
	               ${MCTS_HEADERS}
                   )
    TARGET_LINK_LIBRARIES(${NAME} ${MY_LIBRARY_DEPENDENCIES})
	MESSAGE("-- Adding game: " ${NAME})
ENDMACRO (CREATE_EXAMPLE)

CREATE_EXAMPLE(surakarta)
