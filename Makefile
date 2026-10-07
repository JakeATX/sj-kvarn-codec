# Plain C99 build of the tests (no dependencies). The tree harness has its own script (tests/build_tree_harness.sh).
CC      ?= cc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror
TESTS    = tests/test_vectors tests/test_policy tests/test_kl tests/test_fp16

all: $(TESTS)

tests/%: tests/%.c sj_kvarn.h tests/vector_inputs.h
	$(CC) $(CFLAGS) $< -o $@ -lm

check: all
	./tests/test_fp16
	./tests/test_vectors tests/expected_vectors.txt
	./tests/test_policy
	./tests/test_kl

clean:
	rm -f $(TESTS) tests/tree_harness tests/test_cuda

.PHONY: all check clean
