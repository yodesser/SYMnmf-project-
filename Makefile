# Compiler settings
CC      = gcc
CFLAGS  = -ansi -Wall -Wextra -Werror -pedantic-errors
LDFLAGS = -lm

SRC     = symnmf.c
OBJ     = $(SRC:.c=.o)
RESULT  = symnmf

all: $(RESULT)

$(RESULT): $(OBJ)
	$(CC) $(OBJ) -o $(RESULT) $(LDFLAGS)

%.o: %.c symnmf.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ) $(RESULT)

re: clean all
