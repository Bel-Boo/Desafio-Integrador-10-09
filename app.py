"""
Desafío Integrador - Métodos Numéricos aplicados al Área Financiera
=====================================================================
Aplicación Streamlit que integra:
  - Parte I:   Teoría de errores (absoluto, relativo, porcentual)
  - Parte II:  Aproximación mediante Series de Taylor
  - Parte III: Sistemas de ecuaciones lineales
               (Solución exacta, Factorización LU, Jacobi, Gauss-Seidel)

Autor: Generado con Claude a partir de la guía "Desafío Integrador
       Métodos Numéricos - Área Financiera (90 min)"
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Desafío Integrador - Métodos Numéricos Financieros",
    page_icon="💹",
    layout="wide",
)

# ---------------------------------------------------------------------------
# FUNCIONES NUMÉRICAS
# ---------------------------------------------------------------------------

def calcular_errores(valor_real: float, valor_aprox: float):
    """Devuelve error absoluto, relativo y porcentual."""
    ea = abs(valor_real - valor_aprox)
    er = ea / abs(valor_real) if valor_real != 0 else np.nan
    ep = er * 100
    return ea, er, ep


def taylor_capital(P: float, r: float, t0: float, t: float, orden: int):
    """
    Aproximación de Taylor para C(t) = P * e^(r t) alrededor de t0.
    Como todas las derivadas de una exponencial son P * r^k * e^(r t0),
    se construye el polinomio de Taylor término a término hasta 'orden'.
    Devuelve el valor aproximado y la lista de términos usados.
    """
    from math import factorial, exp
    h = t - t0
    base = P * exp(r * t0)  # = C(t0)
    terminos = []
    aprox = 0.0
    for k in range(orden + 1):
        deriv_k = base * (r ** k)          # C^(k)(t0)
        termino = deriv_k / factorial(k) * (h ** k)
        aprox += termino
        terminos.append(termino)
    return aprox, terminos


def lu_decomposition(A: np.ndarray):
    """Factorización LU con pivoteo parcial: devuelve P, L, U tal que P@A = L@U."""
    n = A.shape[0]
    U = A.astype(float).copy()
    L = np.eye(n)
    P = np.eye(n)
    for k in range(n - 1):
        pivote = np.argmax(np.abs(U[k:, k])) + k
        if pivote != k:
            U[[k, pivote], :] = U[[pivote, k], :]
            P[[k, pivote], :] = P[[pivote, k], :]
            if k > 0:
                L[[k, pivote], :k] = L[[pivote, k], :k]
        for i in range(k + 1, n):
            if U[k, k] == 0:
                continue
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    return P, L, U


def lu_solve(A: np.ndarray, b: np.ndarray):
    """Resuelve Ax = b usando factorización LU con pivoteo parcial."""
    P, L, U = lu_decomposition(A)
    n = len(b)
    Pb = P @ b
    # Sustitución hacia adelante: L y = Pb
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]
    # Sustitución hacia atrás: U x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x, P, L, U


def es_diagonal_dominante(A: np.ndarray) -> bool:
    n = A.shape[0]
    for i in range(n):
        suma_fuera = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        if np.abs(A[i, i]) < suma_fuera:
            return False
    return True


def jacobi(A: np.ndarray, b: np.ndarray, x0=None, tol=1e-6, max_iter=50):
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float).copy()
    historial = [x.copy()]
    for it in range(1, max_iter + 1):
        x_new = np.zeros(n)
        for i in range(n):
            s = A[i, :] @ x - A[i, i] * x[i]
            x_new[i] = (b[i] - s) / A[i, i]
        error = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        historial.append(x.copy())
        if error < tol:
            return x, it, True, historial
    return x, max_iter, False, historial


def gauss_seidel(A: np.ndarray, b: np.ndarray, x0=None, tol=1e-6, max_iter=50):
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float).copy()
    historial = [x.copy()]
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s1 = A[i, :i] @ x[:i]
            s2 = A[i, i + 1:] @ x_old[i + 1:]
            x[i] = (b[i] - s1 - s2) / A[i, i]
        error = np.linalg.norm(x - x_old, ord=np.inf)
        historial.append(x.copy())
        if error < tol:
            return x, it, True, historial
    return x, max_iter, False, historial


# ---------------------------------------------------------------------------
# ESTADO DE SESIÓN (para reutilizar resultados en la conclusión final)
# ---------------------------------------------------------------------------
if "resumen" not in st.session_state:
    st.session_state.resumen = {}

# ---------------------------------------------------------------------------
# ESTILOS GLOBALES (funcionan en tema claro y oscuro, usan variables de Streamlit)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Menú lateral ---- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--gray-30, rgba(128,128,128,0.25));
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0.5rem;
    }
    .menu-titulo {
        font-size: 1.15rem;
        font-weight: 700;
        padding: 0.4rem 0 0.2rem 0;
        margin-bottom: 0.3rem;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.35rem;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 0.55rem 0.8rem;
        width: 100%;
        transition: all 0.15s ease-in-out;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        border-color: var(--primary-color);
        transform: translateX(2px);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label div p {
        font-size: 0.95rem;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background-color: color-mix(in srgb, var(--primary-color) 18%, transparent);
        border-color: var(--primary-color);
    }
    /* ---- Recuadros "cómo resolver" ---- */
    .metodo-box {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-left: 4px solid var(--primary-color);
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.6rem 0 1rem 0;
        background-color: var(--secondary-background-color);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# MENÚ DE NAVEGACIÓN
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="menu-titulo">💹 Navegación</div>', unsafe_allow_html=True)
pagina = st.sidebar.radio(
    "Ir a:",
    [
        "🏠 Inicio y contexto",
        "1️⃣ Parte I - Teoría de errores",
        "2️⃣ Parte II - Aproximación de Taylor",
        "3️⃣ Parte III - Sistemas de ecuaciones",
        "✅ Conclusión integradora",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Desafío Integrador · Métodos Numéricos · Área Financiera\n\n"
    "Teoría de errores · Taylor · LU · Jacobi · Gauss-Seidel"
)

# ---------------------------------------------------------------------------
# PÁGINA: INICIO
# ---------------------------------------------------------------------------
if pagina == "🏠 Inicio y contexto":
    st.title("💹 Desafío Integrador: Métodos Numéricos en el Área Financiera")
    st.subheader("Teoría de errores · Aproximación por Taylor · Sistemas de ecuaciones lineales")

    st.markdown(
        """
### 1. Contexto del desafío
Una pequeña empresa necesita analizar información financiera para tomar decisiones sobre
sus ingresos, costos y distribución de recursos. Los datos provienen de registros y
estimaciones que pueden contener errores de medición, redondeo o aproximación.
El propósito es observar **cómo esos errores pueden afectar los resultados numéricos**.

### 2. Objetivo
Aplicar, de manera sencilla e integrada, la teoría de errores, la aproximación por Taylor
y métodos numéricos para resolver sistemas de ecuaciones lineales en un problema financiero.

### 3. Organización del tiempo (guía original de 90 min)
"""
    )
    tiempos = pd.DataFrame(
        {
            "Parte": ["I", "II", "III", "IV"],
            "Actividad": [
                "Teoría de errores",
                "Aproximación por Taylor",
                "Sistemas de ecuaciones lineales",
                "Conclusión y entrega",
            ],
            "Tiempo sugerido": ["20 min", "25 min", "35 min", "10 min"],
        }
    )
    st.table(tiempos)

    st.info(
        "Usa el menú de la izquierda para recorrer cada parte del desafío. "
        "Todos los datos numéricos pueden editarse desde el teclado."
    )

# ---------------------------------------------------------------------------
# PÁGINA: PARTE I - TEORÍA DE ERRORES
# ---------------------------------------------------------------------------
elif pagina == "1️⃣ Parte I - Teoría de errores":
    st.title("1️⃣ Parte I - Teoría de errores")
    st.markdown(
        """
La empresa compara **valores financieros de referencia** con **valores registrados**.
Calcule el error absoluto, relativo y relativo porcentual para cada dato.

**Fórmulas**
- Error absoluto:  Eₐ = |x_real − x_aproximado|
- Error relativo:  Eᵣ = Eₐ / |x_real|
- Error porcentual: E% = Eᵣ × 100
"""
    )

    with st.expander("📐 Ver método de solución (cómo se resuelve)"):
        st.markdown(
            """
            <div class="metodo-box">

**Pasos para calcular los tres errores de un dato:**

1. Identifica el **valor real** (de referencia) y el **valor aproximado** (registrado).
2. Calcula el **error absoluto**: resta ambos valores y toma el valor absoluto,
   Eₐ = |x_real − x_aproximado|.
3. Calcula el **error relativo** dividiendo el error absoluto entre el valor real,
   Eᵣ = Eₐ / |x_real| (indica qué tan grande es el error en proporción al valor real).
4. Convierte el error relativo a **porcentaje** multiplicando por 100: E% = Eᵣ × 100.

**Ejemplo:** si x_real = 12 500 y x_aproximado = 12 420 →
Eₐ = |12500 − 12420| = 80 → Eᵣ = 80 / 12500 = 0.0064 → E% = 0.64 %.

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("📋 Datos financieros (editables)")
    datos_default = pd.DataFrame(
        {
            "Dato financiero": ["Ingreso diario", "Costo operativo", "Utilidad estimada"],
            "Valor de referencia (Bs)": [12500.0, 7800.0, 4700.0],
            "Valor registrado (Bs)": [12420.0, 7860.0, 4560.0],
        }
    )

    datos = st.data_editor(
        datos_default,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Valor de referencia (Bs)": st.column_config.NumberColumn(format="%.2f"),
            "Valor registrado (Bs)": st.column_config.NumberColumn(format="%.2f"),
        },
        key="tabla_errores",
    )

    if st.button("🧮 Calcular errores", type="primary"):
        filas = []
        for _, fila in datos.iterrows():
            real = float(fila["Valor de referencia (Bs)"])
            aprox = float(fila["Valor registrado (Bs)"])
            ea, er, ep = calcular_errores(real, aprox)
            filas.append(
                {
                    "Dato financiero": fila["Dato financiero"],
                    "Valor real": real,
                    "Valor aproximado": aprox,
                    "Error absoluto (Eₐ)": round(ea, 4),
                    "Error relativo (Eᵣ)": round(er, 6),
                    "Error porcentual (E%)": round(ep, 4),
                }
            )
        resultado = pd.DataFrame(filas)
        st.session_state.resumen["errores"] = resultado

        st.subheader("📊 Resultados")

        st.dataframe(
            resultado.style.format(
                {
                    "Valor real": "{:.2f}",
                    "Valor aproximado": "{:.2f}",
                    "Error absoluto (Eₐ)": "{:.4f}",
                    "Error relativo (Eᵣ)": "{:.6f}",
                    "Error porcentual (E%)": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

        fila_max = resultado.loc[resultado["Error porcentual (E%)"].idxmax()]
        st.warning(
            f"⚠️ El dato con **mayor error porcentual** es **{fila_max['Dato financiero']}** "
            f"con {fila_max['Error porcentual (E%)']:.4f} %."
        )

        st.subheader("📝 Conclusión de la Parte I")
        st.markdown(
            f"""
El error porcentual más alto se presenta en **{fila_max['Dato financiero']}**, lo que indica
que ese valor es el menos confiable de los tres registros frente a su referencia.
En el contexto financiero, un error de este tipo en ingresos o costos puede alterar el
cálculo de la utilidad real de la empresa, llevando a decisiones de inversión, precios o
recorte de gastos basadas en información distorsionada — incluso si el error individual
parece pequeño en términos absolutos.
"""
        )

# ---------------------------------------------------------------------------
# PÁGINA: PARTE II - TAYLOR
# ---------------------------------------------------------------------------
elif pagina == "2️⃣ Parte II - Aproximación de Taylor":
    st.title("2️⃣ Parte II - Aproximación mediante Taylor")
    st.markdown(
        """
Se desea aproximar el crecimiento de un capital usando un modelo simplificado de
**interés continuo**:

$$C(t) = P \\, e^{rt}$$

Por defecto: P = 10 000, r = 0.05, expandiendo alrededor de **t₀ = 1 año** para estimar
el capital en **t = 1.5 años** (valores de la guía original, editables abajo).
"""
    )

    with st.expander("📐 Ver método de solución (cómo se resuelve)"):
        st.markdown(
            """
            <div class="metodo-box">

**Pasos para aproximar C(t) con la Serie de Taylor alrededor de t₀:**

1. Escribe el polinomio de Taylor de orden *n*:
   C(t) ≈ C(t₀) + C'(t₀)(t−t₀) + C''(t₀)/2!·(t−t₀)² + ... + C⁽ⁿ⁾(t₀)/n!·(t−t₀)ⁿ
2. Como C(t) = P·e^(rt), todas sus derivadas son C⁽ᵏ⁾(t) = P·rᵏ·e^(rt), así que
   C⁽ᵏ⁾(t₀) = P·rᵏ·e^(r·t₀).
3. Calcula h = t − t₀ y arma cada término: C⁽ᵏ⁾(t₀)/k! · hᵏ.
4. Suma los términos desde k = 0 hasta el orden elegido para obtener la aproximación.
5. Compara contra el valor exacto C(t) = P·e^(rt) para medir el error de truncamiento.

**A mayor orden, más términos de la serie se incluyen y menor es el error.**

            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        P = st.number_input("Capital inicial P (Bs)", value=10000.0, step=100.0)
    with col2:
        r = st.number_input("Tasa continua r", value=0.05, step=0.01, format="%.4f")
    with col3:
        t0 = st.number_input("Punto de expansión t₀ (años)", value=1.0, step=0.1)
    with col4:
        t = st.number_input("Punto a estimar t (años)", value=1.5, step=0.1)

    if st.button("🧮 Calcular aproximaciones de Taylor", type="primary"):
        from math import exp

        valor_exacto = P * exp(r * t)

        filas = []
        for orden in (1, 2, 3):
            aprox, _ = taylor_capital(P, r, t0, t, orden)
            ea, er, ep = calcular_errores(valor_exacto, aprox)
            filas.append(
                {
                    "Orden": orden,
                    "Aproximación C(t)": round(aprox, 4),
                    "Error absoluto": round(ea, 6),
                    "Error porcentual (%)": round(ep, 6),
                }
            )
        tabla_taylor = pd.DataFrame(filas)
        st.session_state.resumen["taylor"] = tabla_taylor
        st.session_state.resumen["valor_exacto_taylor"] = valor_exacto

        st.subheader("📊 Comparación de aproximaciones")
        st.metric("Valor exacto C(t)", f"{valor_exacto:,.4f}")
        st.dataframe(tabla_taylor, use_container_width=True)

        mejor = tabla_taylor.loc[tabla_taylor["Error absoluto"].idxmin()]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=[f"Orden {o}" for o in tabla_taylor["Orden"]],
                   y=tabla_taylor["Aproximación C(t)"], name="Aproximación Taylor")
        )
        fig.add_hline(y=valor_exacto, line_dash="dash", line_color="red",
                       annotation_text="Valor exacto")
        fig.update_layout(title="Aproximaciones de Taylor vs valor exacto",
                           yaxis_title="Capital C(t)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📝 Conclusión de la Parte II")
        st.markdown(
            f"""
El **Taylor de orden {int(mejor['Orden'])}** produjo la mejor aproximación, con un error
absoluto de {mejor['Error absoluto']:.6f}. Como es de esperarse, entre más términos del
polinomio se incluyen, más se acerca la aproximación al valor exacto, porque se capturan
mejor la curvatura y variaciones de la función exponencial cerca de t₀. En un contexto
financiero, esto muestra que los modelos linealizados (orden 1) pueden ser suficientes
para horizontes cortos, pero subestiman o sobreestiman el crecimiento real del capital
si el intervalo (t − t₀) crece.
"""
        )

# ---------------------------------------------------------------------------
# PÁGINA: PARTE III - SISTEMAS DE ECUACIONES
# ---------------------------------------------------------------------------
elif pagina == "3️⃣ Parte III - Sistemas de ecuaciones":
    st.title("3️⃣ Parte III - Sistemas de ecuaciones lineales")
    st.markdown(
        """
La empresa distribuye recursos entre áreas. Las incógnitas representan montos en
**miles de bolivianos**:  x₁ = operaciones · x₂ = comercialización · x₃ = tecnología
(y x₄ = un área adicional, si eliges una matriz 4x4).
"""
    )

    with st.expander("📐 Ver métodos de solución (cómo se resuelve cada uno)"):
        st.markdown(
            """
            <div class="metodo-box">

**Solución exacta (eliminación gaussiana):**
Reduce la matriz aumentada [A | b] a forma escalonada usando operaciones entre filas
hasta obtener un sistema triangular, y luego se despeja cada incógnita por sustitución
hacia atrás. Da la solución exacta cuando A no es singular.

**Factorización LU (con pivoteo parcial):**
1. Descompone A en P·A = L·U (L triangular inferior, U triangular superior, P de permutación).
2. Resuelve L·y = P·b por **sustitución hacia adelante**.
3. Resuelve U·x = y por **sustitución hacia atrás**.
Es eficiente cuando se debe resolver el sistema para varios vectores b.

**Método de Jacobi (iterativo):**
Despeja cada xᵢ de su propia ecuación usando los valores de la iteración **anterior**:
xᵢ⁽ᵏ⁺¹⁾ = (bᵢ − Σⱼ≠ᵢ aᵢⱼ·xⱼ⁽ᵏ⁾) / aᵢᵢ.
Se repite hasta que el cambio entre iteraciones sea menor a la tolerancia. Requiere que
la matriz sea diagonal dominante para garantizar convergencia.

**Método de Gauss-Seidel (iterativo):**
Igual que Jacobi, pero usa los valores **ya actualizados** de xⱼ dentro de la misma
iteración en lugar de esperar a la siguiente, por lo que normalmente converge más rápido.

            </div>
            """,
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns(2)
    with col_a:
        tam = st.selectbox("Tamaño de la matriz", ["3x3", "4x4"])
    with col_b:
        escenario = st.selectbox(
            "Escenario preestablecido",
            ["Caso ideal", "Caso bajo estrés", "Caso mal condicionado", "Personalizado"]
            if tam == "3x3" else ["Personalizado"],
        )

    n = 3 if tam == "3x3" else 4

    # ----- Escenarios predefinidos de la guía (solo 3x3) -----
    if tam == "3x3" and escenario == "Caso ideal":
        A_default = np.array([[10, 2, 1], [1, 9, 2], [2, 1, 8]], dtype=float)
        b_default = np.array([29, 25, 28], dtype=float)
    elif tam == "3x3" and escenario == "Caso bajo estrés":
        A_default = np.array([[10, 2, 1], [1, 9, 2], [2, 1, 8]], dtype=float)
        b_default = np.array([29.5, 24.5, 28.4], dtype=float)
    elif tam == "3x3" and escenario == "Caso mal condicionado":
        A_default = np.array([[1, 1, 1], [1.001, 1, 1], [1, 1.001, 1]], dtype=float)
        b_default = np.array([6, 6.001, 6.002], dtype=float)
    else:
        A_default = np.eye(n) * 5 + 1
        b_default = np.ones(n) * 10

    st.subheader("📋 Matriz A y vector b (editables)")
    cols_nombres = [f"x{i+1}" for i in range(n)] + ["b"]
    tabla_default = pd.DataFrame(
        np.hstack([A_default, b_default.reshape(-1, 1)]), columns=cols_nombres
    )
    tabla_sistema = st.data_editor(
        tabla_default,
        use_container_width=True,
        key=f"tabla_sistema_{tam}_{escenario}",
        column_config={c: st.column_config.NumberColumn(format="%.4f") for c in cols_nombres},
    )

    A = tabla_sistema[[f"x{i+1}" for i in range(n)]].to_numpy(dtype=float)
    b = tabla_sistema["b"].to_numpy(dtype=float)

    if not es_diagonal_dominante(A):
        st.warning(
            "⚠️ La matriz **no** es estrictamente diagonal dominante: Jacobi y "
            "Gauss-Seidel podrían no converger o converger lentamente."
        )

    st.subheader("⚙️ Parámetros de los métodos iterativos")
    c1, c2, c3 = st.columns(3)
    with c1:
        tol = st.number_input("Tolerancia (TOL)", value=1e-6, format="%.1e")
    with c2:
        max_iter = st.number_input("Máximo de iteraciones", value=50, step=1, min_value=1)
    with c3:
        x0_str = st.text_input("Vector inicial x⁽⁰⁾ (separado por comas)",
                                value=",".join(["0"] * n))
    try:
        x0 = np.array([float(v) for v in x0_str.split(",")])
        if len(x0) != n:
            x0 = np.zeros(n)
    except ValueError:
        x0 = np.zeros(n)

    metodos = st.multiselect(
        "Métodos a calcular",
        ["Solución exacta", "Factorización LU", "Jacobi", "Gauss-Seidel"],
        default=["Solución exacta", "Factorización LU", "Jacobi", "Gauss-Seidel"],
    )

    if st.button("🧮 Resolver sistema", type="primary"):
        resultados = []

        if "Solución exacta" in metodos:
            try:
                x_exacta = np.linalg.solve(A, b)
                resultados.append(
                    {"Método": "Solución exacta (numpy)", "x": x_exacta,
                     "Iteraciones": "-", "Convergió": "-"}
                )
            except np.linalg.LinAlgError:
                st.error("La matriz es singular: no tiene solución exacta única.")
                x_exacta = None

        if "Factorización LU" in metodos:
            try:
                x_lu, P, L, U = lu_solve(A, b)
                resultados.append(
                    {"Método": "Factorización LU", "x": x_lu,
                     "Iteraciones": "-", "Convergió": "-"}
                )
                with st.expander("Ver matrices L y U (factorización LU)"):
                    st.write("**P (permutación)**")
                    st.dataframe(pd.DataFrame(P))
                    st.write("**L**")
                    st.dataframe(pd.DataFrame(L).round(6))
                    st.write("**U**")
                    st.dataframe(pd.DataFrame(U).round(6))
            except Exception as e:
                st.error(f"Error en LU: {e}")

        if "Jacobi" in metodos:
            x_j, it_j, conv_j, _ = jacobi(A, b, x0, tol, int(max_iter))
            resultados.append(
                {"Método": "Jacobi", "x": x_j, "Iteraciones": it_j,
                 "Convergió": "Sí" if conv_j else "No"}
            )

        if "Gauss-Seidel" in metodos:
            x_gs, it_gs, conv_gs, _ = gauss_seidel(A, b, x0, tol, int(max_iter))
            resultados.append(
                {"Método": "Gauss-Seidel", "x": x_gs, "Iteraciones": it_gs,
                 "Convergió": "Sí" if conv_gs else "No"}
            )

        st.subheader("📊 Resultados")
        tabla_res = pd.DataFrame(
            [
                {
                    "Método": r["Método"],
                    **{f"x{i+1}": round(r["x"][i], 6) for i in range(n)},
                    "Iteraciones": r["Iteraciones"],
                    "Convergió": r["Convergió"],
                }
                for r in resultados
            ]
        )
        st.dataframe(tabla_res, use_container_width=True)
        st.session_state.resumen["sistema"] = tabla_res
        st.session_state.resumen["sistema_escenario"] = f"{tam} - {escenario}"

        st.subheader("📝 Conclusión de la Parte III")
        if escenario == "Caso mal condicionado":
            st.markdown(
                """
En el **caso mal condicionado**, las tres ecuaciones son casi paralelas (coeficientes muy
parecidos entre sí). Esto hace que el sistema sea **muy sensible** a pequeños cambios en
los términos independientes: una variación mínima en un dato (de 6.002 a 6.003, por
ejemplo) puede producir un cambio proporcionalmente grande en la solución. Financieramente,
esto representa un escenario de alto riesgo: estimaciones casi idénticas entre áreas hacen
que la asignación de recursos calculada sea poco confiable ante cualquier error de medición.
"""
            )
        elif escenario == "Caso bajo estrés":
            st.markdown(
                """
En el **caso bajo estrés**, el vector de resultados cambia ligeramente respecto al caso
ideal, pero la matriz de coeficientes (bien condicionada y diagonal dominante) se mantiene.
Por eso la solución varía solo de forma proporcional y controlada, y el número de
iteraciones de Jacobi/Gauss-Seidel se mantiene similar: el sistema es **estable** ante
pequeñas perturbaciones.
"""
            )
        else:
            st.markdown(
                """
En el **caso ideal**, al ser la matriz diagonal dominante, tanto Jacobi como Gauss-Seidel
convergen en pocas iteraciones hacia la misma solución que entrega el método exacto y la
factorización LU, confirmando la consistencia de los cuatro enfoques. Gauss-Seidel
normalmente converge en menos iteraciones que Jacobi porque usa los valores actualizados
de manera inmediata dentro de la misma iteración.
"""
            )

    st.markdown("---")
    st.caption(
        "💡 Tip: cambia el escenario a **Caso mal condicionado**, resuelve con LU, "
        "luego edita el último valor de b (6.002 → 6.003) y vuelve a resolver para "
        "comparar ambas soluciones, tal como pide la guía."
    )

# ---------------------------------------------------------------------------
# PÁGINA: CONCLUSIÓN INTEGRADORA
# ---------------------------------------------------------------------------
elif pagina == "✅ Conclusión integradora":
    st.title("✅ Conclusión integradora")
    st.markdown(
        """
**Pregunta guía:** ¿Puede un error pequeño en un dato financiero producir un cambio
importante en el resultado final? Relaciona tu respuesta con los errores calculados,
la aproximación de Taylor y el sistema mal condicionado.
"""
    )

    resumen = st.session_state.resumen
    if not resumen:
        st.info(
            "Aún no has generado resultados. Visita las Partes I, II y III desde el "
            "menú para calcular los datos que resumirá esta sección."
        )
    else:
        if "errores" in resumen:
            st.subheader("Resumen · Parte I")
            st.dataframe(resumen["errores"], use_container_width=True)
        if "taylor" in resumen:
            st.subheader("Resumen · Parte II")
            st.dataframe(resumen["taylor"], use_container_width=True)
        if "sistema" in resumen:
            st.subheader(f"Resumen · Parte III ({resumen.get('sistema_escenario', '')})")
            st.dataframe(resumen["sistema"], use_container_width=True)

    st.markdown("### 📝 Conclusión final (máximo 5 líneas)")
    st.markdown(
        """
Sí: un error pequeño en un dato financiero puede propagarse y producir un cambio
importante en el resultado final. Los errores porcentuales de la Parte I ya son
relevantes en términos relativos; la aproximación de Taylor confirma que truncar una
serie introduce un error que crece con la distancia al punto de expansión; y el sistema
mal condicionado de la Parte III muestra que, con ecuaciones casi linealmente
dependientes, ese error pequeño se amplifica en la solución. Por eso la empresa debe
tratar sus datos financieros con controles de precisión adecuados antes de usarlos en
decisiones de asignación de recursos.
"""
    )
