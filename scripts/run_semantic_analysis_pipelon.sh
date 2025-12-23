echo "================================================"
echo "PIPELINE: ANÁLISIS DE SESGOS SEMÁNTICOS"
echo "================================================"
echo ""

echo "Paso 1: Ejecutar experimento comparativo..."
python scripts/run_experiment_semantic_comparison.py

if [ $? -ne 0 ]; then
    echo " Error en experimento"
    exit 1
fi

echo ""
echo "Paso 2: Generar análisis cualitativo..."
python scripts/qualitative_comparison.py

echo ""
echo "Paso 3: Abrir notebook de visualización..."
echo "   Ejecuta: jupyter notebook examples/02_semantic_comparison.ipynb"

echo ""
echo "================================================"
echo " Pipeline completado"
echo "================================================"