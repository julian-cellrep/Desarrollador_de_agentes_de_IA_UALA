# Clase 1 — Demo con GPT-2

Cuaderno para ver, sobre [GPT-2 small](https://huggingface.co/gpt2) (124M, Hugging Face), lo que la clase plantea en abstracto:

1. El tokenizador byte-level BPE.
2. La matriz de atención y las dimensiones de las salidas del Transformer.
3. La distribución de probabilidad del próximo token.
4. Cómo cambia esa distribución al muestrear con distintas temperaturas.

El modelo corre en CPU. No hace falta GPU ni credencial de ningún proveedor. La primera ejecución descarga los pesos (~500 MB).

## Entorno

Desde esta carpeta (`clase-01-fundamentos-llms/`):

```bash
conda env create -f environment.yml
conda activate clase-01-gpt2
python -m ipykernel install --user --name clase-01-gpt2 --display-name "clase-01-gpt2"
```

Abrí `clase-01-gpt2.ipynb` y elegí el kernel **clase-01-gpt2**. Ejecutá las celdas en orden.

Para recrear el entorno desde cero:

```bash
conda env remove -n clase-01-gpt2
conda env create -f environment.yml
```

## Qué usa cada sección

| Sección | Idea |
|---|---|
| 1 | `GPT2Tokenizer`: piezas BPE, el marcador `Ġ` de espacio, y cuántos tokens consume el español frente al inglés |
| 2 | Una pasada de `GPT2LMHeadModel` con atención eager: formas `(batch, cabezas, seq, seq)`, `(batch, seq, 768)` y `(batch, seq, 50257)` |
| 3 | Softmax de los logits del último token (T = 1) y una continuación greedy |
| 4 | Los mismos logits a T = 0.1, 0.5, 1.0, 1.5 y 2.0, más tres continuaciones muestreadas por temperatura |
