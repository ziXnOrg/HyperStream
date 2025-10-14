// Minimal C test harness for HyperStream C API (Increment 1 scaffolding)
// Purpose: Ensure the header compiles in C and the shared library links.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "hyperstream/hyperstream.h"

int main(void) {
  /* Smoke: enumerate a couple of constants to ensure header is visible */
  volatile int zero = (int)HS_OK;
  if (zero != 0) {
    fprintf(stderr, "Unexpected status enum value\n");
    return 1;
  }
  /* No API calls yet; linking should succeed with empty TU implementation. */
  printf("c_api_tests: header available, build ok\n");
  return 0;
}
