.PHONY: all clean build test help

FPC = fpc
FPCFLAGS = -O2 -Mobjfpc -Scghi

# Build targets
all: build

build: nraay_tensor_ref

nraay_tensor_ref: nraay_tensor_ref.pas buffer.pas tensor_array.pas \
                   descriptor.pas ownership.pas governance.pas \
                   materialize.pas tensor.pas validation.pas tests.pas
	$(FPC) $(FPCFLAGS) -onraay_tensor_ref nraay_tensor_ref.pas

clean:
	rm -f nraay_tensor_ref
	rm -f *.o *.ppu
	rm -f ppas.sh

test: build
	./nraay_tensor_ref

help:
	@echo "NraayTensor Pascal Reference Implementation"
	@echo ""
	@echo "Targets:"
	@echo "  make build   - Compile reference implementation"
	@echo "  make test    - Build and run test suite"
	@echo "  make clean   - Remove build artifacts"
	@echo "  make help    - Display this message"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Build executable"
	@echo "  make test     - Build and run tests (exit 0 = pass, 1 = fail)"
	@echo ""
	@echo "FPC version required: 3.2.0 or later"
	@echo "FPC switches: -O2 (optimize), -Mobjfpc (object mode), -Scghi (output)"
