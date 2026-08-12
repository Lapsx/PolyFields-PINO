#!/bin/bash

# Navega para o diretório onde o script está localizado
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "========================================="
echo " Iniciando o FNO Polymer Sandbox..."
echo "========================================="

# (Opcional) Tenta fechar instâncias anteriores que possam ter ficado presas na porta 8000
pkill -f "uvicorn main:app" 2>/dev/null

# Programa a abertura do navegador (espera 2 segundos para o backend subir)
(sleep 2 && xdg-open "file://$DIR/frontend/index.html") &

# Inicia o backend na mesma janela
cd backend
echo "Iniciando o servidor de IA (FastAPI)..."
echo "Para encerrar a aplicação, pressione Ctrl+C nesta janela."
uvicorn main:app
