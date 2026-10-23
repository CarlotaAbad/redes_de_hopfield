import gzip
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

SEMILLA = 42
np.random.seed(SEMILLA)
rng = np.random.default_rng(SEMILLA)

N = 784
os.makedirs("figuras", exist_ok=True)
resultados = {}

# 1. CARGAR MNIST
def cargar_mnist():
    try:
        with gzip.open("mnist/train-images-idx3-ubyte.gz") as f:
            X = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        with gzip.open("mnist/train-labels-idx1-ubyte.gz") as f:
            y = np.frombuffer(f.read(), np.uint8, offset=8).astype(int)
        print("MNIST leído de la carpeta mnist/")
    except FileNotFoundError:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X, y = X[:60000], y[:60000].astype(int)
        print("MNIST descargado de OpenML")
    return X, y

# 2. FUNCIONES DE LA RED DE HOPFIELD
def entrenar_hebb(patrones):
    """Regla de Hebb: w_ij = (1/N) * suma_s x_i^(s) x_j^(s), con w_ii = 0.
    'patrones' tiene un patrón (vector de +1/-1) en cada fila."""
    W = np.zeros((N, N))
    for x in patrones:
        W += np.outer(x, x)
    W = W / N
    np.fill_diagonal(W, 0)
    return W


def signo(h):
    return np.where(h >= 0, 1, -1)


def energia(W, x):
    return -0.5 * x @ W @ x


def dinamica_sincrona(W, x, max_iter=100):
    x = x.copy()
    energias = [energia(W, x)]
    anterior = None
    for it in range(max_iter):
        nuevo = signo(W @ x)
        energias.append(energia(W, nuevo))
        if np.array_equal(nuevo, x):
            return nuevo, energias, "converge"
        if anterior is not None and np.array_equal(nuevo, anterior):
            return nuevo, energias, "ciclo"
        anterior = x
        x = nuevo
    return x, energias, "max_iter"


def dinamica_asincrona(W, x, max_barridos=50, guardar_energia=False, foto_cada=None):
    x = x.copy()
    energias = [energia(W, x)] if guardar_energia else []
    fotos = [x.copy()] if foto_cada else []
    paso = 0
    for barrido in range(1, max_barridos + 1):
        cambios = 0
        for i in range(N):
            nuevo = 1 if W[i] @ x >= 0 else -1
            if nuevo != x[i]:
                x[i] = nuevo
                cambios += 1
            paso += 1
            if guardar_energia:
                energias.append(energia(W, x))
            if foto_cada and paso % foto_cada == 0:
                fotos.append(x.copy())
        if cambios == 0:
            break
    return x, energias, barrido, fotos


def hamming(a, b):
    return np.sum(a != b)


def clasificar(estado, patrones, etiquetas):
    distancias = [hamming(estado, p) for p in patrones]
    return etiquetas[np.argmin(distancias)]


def meter_ruido(X, p):
    X = X.copy()
    voltear = rng.random(X.shape) < p
    X[voltear] = -X[voltear]
    return X


def evaluar(W, X_entrada, modo="asincrona"):
    predicciones, finales = [], []
    for x in X_entrada:
        if modo == "sincrona":
            final, _, _ = dinamica_sincrona(W, x)
        elif modo == "asincrona":
            final, _, _, _ = dinamica_asincrona(W, x)
        else:
            final = x
        predicciones.append(clasificar(final, X_train, y_train))
        finales.append(final)
    return np.array(predicciones), np.array(finales)


def dibujar(ax, v, titulo=""):
    ax.imshow(v.reshape(28, 28), cmap="gray_r", vmin=-1, vmax=1)
    ax.set_title(titulo, fontsize=8)
    ax.axis("off")

# 3. SUBCONJUNTO DE 100 IMÁGENES Y PREPROCESAMIENTO
X_todo, y_todo = cargar_mnist()

indices = []
for digito in range(10):
    posiciones = np.where(y_todo == digito)[0]
    elegidas = np.random.choice(posiciones, 10, replace=False)
    indices.extend(elegidas)
indices = np.array(indices)

X_gris = X_todo[indices] / 255.0
y = y_todo[indices]
X = np.where(X_gris > 0.5, 1, -1)
print("Subconjunto:", X.shape, "| imágenes por dígito:", np.bincount(y))
resultados["indices_mnist"] = indices.tolist()
resultados["pixeles_trazo"] = float(np.mean(X == 1))

fig, ejes = plt.subplots(2, 10, figsize=(10, 2.4))
for d in range(10):
    k = np.where(y == d)[0][0]
    ejes[0, d].imshow(X_gris[k].reshape(28, 28), cmap="gray_r")
    ejes[0, d].axis("off")
    dibujar(ejes[1, d], X[k])
ejes[0, 0].set_title("Original (gris)", fontsize=8, loc="left")
ejes[1, 0].set_title("Binarizada (±1)", fontsize=8, loc="left")
plt.tight_layout()
plt.savefig("figuras/fig1_preprocesamiento.png", dpi=150)
plt.close()

# 4. DIVISIÓN ESTRATIFICADA 70 / 30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEMILLA)
print("Train:", len(y_train), "por dígito", np.bincount(y_train))
print("Test: ", len(y_test), "por dígito", np.bincount(y_test))
resultados["train_por_digito"] = np.bincount(y_train).tolist()
resultados["test_por_digito"] = np.bincount(y_test).tolist()

# 5. ENTRENAMIENTO CON LA REGLA DE HEBB
W = entrenar_hebb(X_train)
print("\nW es simétrica:", np.allclose(W, W.T), "| diagonal a cero:", np.all(np.diag(W) == 0))

estables = sum(np.array_equal(signo(W @ p), p) for p in X_train)
bits_mal = np.mean([np.sum(signo(W @ p) != p) for p in X_train])
print(f"Patrones guardados que son estables: {estables} de {len(X_train)}")
print(f"Al actualizar un patrón guardado cambian de media {bits_mal:.1f} píxeles")
resultados["hebb_estables"] = int(estables)
resultados["hebb_bits_cambian"] = float(bits_mal)

# 6. CLASIFICACIÓN SIN RUIDO (síncrona y asíncrona)
print("\n--- Clasificación con Hebb, sin ruido ---")
fig, ejes = plt.subplots(1, 2, figsize=(10, 4.2))
for eje, modo in zip(ejes, ["sincrona", "asincrona"]):
    pred, finales = evaluar(W, X_test, modo)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=range(10))
    distintos = len(np.unique(finales, axis=0))
    print(f"{modo}: accuracy = {acc:.2f}, estados finales distintos = {distintos}")
    print(cm)
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=eje, cmap="Blues", colorbar=False)
    eje.set_title(f"Dinámica {modo} (accuracy = {acc:.0%})")
    eje.set_xlabel("Clase predicha")
    eje.set_ylabel("Clase real")
    resultados[f"hebb_{modo}"] = {"accuracy": float(acc), "matriz": cm.tolist(),
                                  "estados_distintos": int(distintos)}
plt.tight_layout()
plt.savefig("figuras/fig2_matrices_confusion_hebb.png", dpi=150)
plt.close()

# 7. ENERGÍA Y EVOLUCIÓN DE LA RED (una imagen de test con 10 % de ruido)
x0 = meter_ruido(X_test[:1], 0.10)[0]
_, E_asin, barridos, _ = dinamica_asincrona(W, x0, guardar_energia=True)
_, E_sin, estado_sin = dinamica_sincrona(W, x0)
sube = np.any(np.diff(E_asin) > 1e-9)
print(f"\nEnergía asíncrona: {E_asin[0]:.1f} -> {E_asin[-1]:.1f} en {barridos} barridos"
      f" (¿sube alguna vez? {sube})")
print(f"Energía síncrona:  {E_sin[0]:.1f} -> {E_sin[-1]:.1f} ({estado_sin}, {len(E_sin) - 1} pasos)")
resultados["energia"] = {"digito": int(y_test[0]), "inicial": float(E_asin[0]),
                         "final": float(E_asin[-1]), "barridos": int(barridos),
                         "sube_alguna_vez": bool(sube), "pasos_sincrona": len(E_sin) - 1,
                         "estado_sincrona": estado_sin}

fig, ejes = plt.subplots(1, 2, figsize=(10, 3.6))
ejes[0].plot(np.arange(len(E_asin)) / N, E_asin)
ejes[0].set_xlabel("Barridos (1 barrido = 784 neuronas actualizadas)")
ejes[0].set_ylabel("Energía E")
ejes[0].set_title(f"Asíncrona: converge en {barridos} barridos")
ejes[0].grid(alpha=0.3)
ejes[1].plot(E_sin, "o-", color="tab:orange")
ejes[1].set_xlabel("Iteración síncrona")
ejes[1].set_ylabel("Energía E")
ejes[1].set_title(f"Síncrona: {estado_sin} en {len(E_sin) - 1} pasos")
ejes[1].set_xticks(range(len(E_sin)))
ejes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figuras/fig3_energia.png", dpi=150)
plt.close()

_, _, _, fotos = dinamica_asincrona(W, x0, max_barridos=1, foto_cada=112)
fig, ejes = plt.subplots(1, len(fotos), figsize=(1.4 * len(fotos), 1.8))
for k, (eje, foto) in enumerate(zip(ejes, fotos)):
    dibujar(eje, foto, f"{k * 112} neuronas")
plt.suptitle("Evolución del estado durante el primer barrido (Hebb)", fontsize=10)
plt.tight_layout()
plt.savefig("figuras/fig4_evolucion_estado.png", dpi=150)
plt.close()

# 8. ¿POR QUÉ FALLA? Estado final y parecido entre los dígitos
_, finales_hebb = evaluar(W, X_test, "asincrona")
atractor = finales_hebb[0]
media = signo(X_train.mean(axis=0))
cercano = np.argmin([hamming(atractor, p) for p in X_train])
resultados["atractor"] = {
    "pixeles_trazo": float(np.mean(atractor == 1)),
    "hamming_a_media": int(hamming(atractor, media)),
    "hamming_a_mas_cercano": int(hamming(atractor, X_train[cercano])),
    "digito_mas_cercano": int(y_train[cercano]),
    "energia": float(energia(W, atractor)),
    "energia_media_patrones": float(np.mean([energia(W, p) for p in X_train])),
}
print("\nEstado final común:", resultados["atractor"])

fig, ejes = plt.subplots(1, 3, figsize=(6, 2.4))
dibujar(ejes[0], atractor, "Estado final de la red")
dibujar(ejes[1], media, "Signo de la media de los 70")
dibujar(ejes[2], X_train[cercano], f"Patrón más cercano (un {y_train[cercano]})")
plt.tight_layout()
plt.savefig("figuras/fig5_estado_espurio.png", dpi=150)
plt.close()

solap_mnist = (X_train @ X_train.T / N)[~np.eye(70, dtype=bool)]
aleatorios = np.where(rng.random((70, N)) < 0.5, 1, -1)
solap_aleat = (aleatorios @ aleatorios.T / N)[~np.eye(70, dtype=bool)]
print(f"Solapamiento medio: MNIST {solap_mnist.mean():.2f} | aleatorios {np.abs(solap_aleat).mean():.2f}")
resultados["solapamiento_mnist"] = float(solap_mnist.mean())
resultados["solapamiento_aleatorio"] = float(np.abs(solap_aleat).mean())

plt.figure(figsize=(7, 3.4))
plt.hist(solap_aleat, bins=30, alpha=0.7, label="Patrones aleatorios")
plt.hist(solap_mnist, bins=30, alpha=0.7, label="Dígitos MNIST")
plt.xlabel("Solapamiento m entre dos patrones")
plt.ylabel("Número de parejas")
plt.title("Los dígitos se parecen mucho más entre sí que unos patrones al azar")
plt.legend()
plt.tight_layout()
plt.savefig("figuras/fig6_solapamientos.png", dpi=150)
plt.close()

# 9. CAPACIDAD: ¿cuántos patrones aguanta la red?
print("\n--- Capacidad (porcentaje de patrones que siguen estables) ---")
lista_P = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140]
cap_aleat, cap_mnist = [], []
for P in lista_P:
    fracciones = []
    for rep in range(3):
        A = np.where(rng.random((P, N)) < 0.5, 1, -1)
        WA = entrenar_hebb(A)
        fracciones.append(np.mean([np.array_equal(signo(WA @ a), a) for a in A]))
    cap_aleat.append(float(np.mean(fracciones)))
    if P <= 70:
        WM = entrenar_hebb(X_train[:P])
        cap_mnist.append(float(np.mean([np.array_equal(signo(WM @ a), a) for a in X_train[:P]])))
    print(f"P = {P:3d}: aleatorios {cap_aleat[-1]:.2f}" + (f" | MNIST {cap_mnist[-1]:.2f}" if P <= 70 else ""))
resultados["capacidad"] = {"P": lista_P, "aleatorios": cap_aleat, "mnist": cap_mnist}

plt.figure(figsize=(7, 3.8))
plt.plot(lista_P, np.array(cap_aleat) * 100, "o-", label="Patrones aleatorios")
plt.plot(lista_P[:len(cap_mnist)], np.array(cap_mnist) * 100, "s-", label="Dígitos MNIST")
plt.axvline(0.138 * N, color="red", ls="--", label="0.138·N ≈ 108 (teoría)")
plt.axvline(70, color="gray", ls=":", label="P = 70 (nuestro caso)")
plt.xlabel("Número de patrones guardados P")
plt.ylabel("% de patrones estables")
plt.title("Capacidad de la red con la regla de Hebb (N = 784)")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figuras/fig7_capacidad.png", dpi=150)
plt.close()

# 10. MEJORA QUE PROBAMOS: REGLA DE PROYECCIÓN (pseudoinversa)
Xi = X_train.T
W_proy = Xi @ np.linalg.pinv(Xi)
np.fill_diagonal(W_proy, 0)
estables_proy = sum(np.array_equal(signo(W_proy @ p), p) for p in X_train)
pred_proy, _ = evaluar(W_proy, X_test, "asincrona")
acc_proy = accuracy_score(y_test, pred_proy)
cm_proy = confusion_matrix(y_test, pred_proy, labels=range(10))
print(f"\nRegla de proyección: {estables_proy}/70 estables, accuracy = {acc_proy:.2f}")
print(cm_proy)
resultados["proyeccion"] = {"estables": int(estables_proy), "accuracy": float(acc_proy),
                            "matriz": cm_proy.tolist()}

fig, eje = plt.subplots(figsize=(4.6, 4.2))
ConfusionMatrixDisplay(cm_proy, display_labels=range(10)).plot(ax=eje, cmap="Blues", colorbar=False)
eje.set_title(f"Regla de proyección (accuracy = {acc_proy:.0%})")
eje.set_xlabel("Clase predicha")
eje.set_ylabel("Clase real")
plt.tight_layout()
plt.savefig("figuras/fig8_matriz_confusion_proyeccion.png", dpi=150)
plt.close()

# 11. EJEMPLOS DE RECUPERACIÓN (10 % de ruido): Hebb vs proyección
ejemplos = [np.where(y_test == d)[0][0] for d in [0, 3, 4, 7, 9]]
X_ej = meter_ruido(X_test[ejemplos], 0.10)
_, fin_hebb = evaluar(W, X_ej, "asincrona")
_, fin_proy = evaluar(W_proy, X_ej, "asincrona")
fig, ejes = plt.subplots(len(ejemplos), 4, figsize=(6, 1.6 * len(ejemplos)))
for f, k in enumerate(ejemplos):
    dibujar(ejes[f, 0], X_test[k], f"Original ({y_test[k]})")
    dibujar(ejes[f, 1], X_ej[f], "Con 10 % de ruido")
    dibujar(ejes[f, 2], fin_hebb[f], f"Hebb -> {clasificar(fin_hebb[f], X_train, y_train)}")
    dibujar(ejes[f, 3], fin_proy[f], f"Proyección -> {clasificar(fin_proy[f], X_train, y_train)}")
plt.tight_layout()
plt.savefig("figuras/fig9_ejemplos_recuperacion.png", dpi=150)
plt.close()

# 12. ACCURACY CON RUIDO (0 %, 5 %, 10 %, 20 %), repitiendo 5 veces el ruido
print("\n--- Accuracy con ruido (media de 5 repeticiones) ---")
niveles = [0.0, 0.05, 0.10, 0.20]
metodos = {"Hebb": (W, "asincrona"), "Proyección": (W_proy, "asincrona"),
           "Sin red (comparar directamente)": (None, "sin red")}
tabla = {m: {"media": [], "desv": []} for m in metodos}
for p in niveles:
    repeticiones = 1 if p == 0 else 5
    accs = {m: [] for m in metodos}
    for rep in range(repeticiones):
        X_ruido = meter_ruido(X_test, p)
        for nombre, (Wm, modo) in metodos.items():
            pred, _ = evaluar(Wm, X_ruido, modo)
            accs[nombre].append(accuracy_score(y_test, pred))
    for nombre in metodos:
        tabla[nombre]["media"].append(float(np.mean(accs[nombre])))
        tabla[nombre]["desv"].append(float(np.std(accs[nombre])))
    print(f"ruido {p:.0%}: " + " | ".join(f"{n}: {np.mean(a):.2f}" for n, a in accs.items()))
resultados["ruido"] = {"niveles": niveles, "tabla": tabla}

plt.figure(figsize=(7, 3.8))
for nombre, marca in zip(metodos, ["o-", "s-", "^--"]):
    plt.errorbar(np.array(niveles) * 100, np.array(tabla[nombre]["media"]) * 100,
                 yerr=np.array(tabla[nombre]["desv"]) * 100, fmt=marca, capsize=3, label=nombre)
plt.axhline(10, color="gray", ls=":", label="Azar (10 %)")
plt.xticks([0, 5, 10, 20])
plt.xlabel("Ruido (% de píxeles invertidos)")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.title("Accuracy en test según el ruido")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figuras/fig10_accuracy_ruido.png", dpi=150)
plt.close()

# 13. GUARDAR RESULTADOS
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
print("\nListo: figuras en la carpeta 'figuras' y números en resultados.json")