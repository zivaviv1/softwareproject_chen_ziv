CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LDLIBS = -lm
TARGET = symnmf

$(TARGET): symnmf.c symnmf.h
	$(CC) $(CFLAGS) -o $(TARGET) symnmf.c $(LDLIBS)

clean:
	rm -f $(TARGET) symnmfmodule*.so
	rm -rf build __pycache__

.PHONY: clean
