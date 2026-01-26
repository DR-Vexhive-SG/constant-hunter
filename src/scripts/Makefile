# Makefile para demo
CC = nvcc
CFLAGS = -O2 -arch=sm_75
TARGET = constant_hunter_demo
VERSION = demo-1.0

all: $(TARGET)

$(TARGET): constant_hunter_demo.cu
	$(CC) $(CFLAGS) -o $(TARGET) constant_hunter_demo.cu
	@echo "========================================="
	@echo "   VERSIÓN DEMO COMPILADA: $(VERSION)"
	@echo "   (Solo para evaluación técnica)"
	@echo "========================================="

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

# Meta-información para empleadores
info:
	@echo "========================================="
	@echo "   CONSTANT HUNTER - INFORMACIÓN TÉCNICA"
	@echo "========================================="
	@echo "ARQUITECTURA:"
	@echo "   • Paralelización masiva en GPU"
	@echo "   • Memoria jerárquica optimizada"
	@echo "   • Stream processing para datos grandes"
	@echo ""
	@echo "CARACTERÍSTICAS AVANZADAS (versión completa):"
	@echo "   • Multi-stream concurrente"
	@echo "   • Overlap copy/compute"
	@echo "   • Kernel fusion para reducciones"
	@echo "   • Async data prefetching"
	@echo ""
	@echo "CONTACTO PARA CÓDIGO COMPLETO:"
	@echo "   Disponible bajo NDA para entrevistas"
