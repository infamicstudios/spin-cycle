NVCC := nvcc

NVCC_FLAGS := -arch=sm_60

DEBUG_FLAGS := -G -DDEBUG

SRC := spin_cycle.cu

TARGET := spin_cycle
DEBUG_TARGET := $(TARGET)_debug

all: $(TARGET)

debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: $(DEBUG_TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(DEBUG_TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGET) $(DEBUG_TARGET)

.PHONY: all debug clean
