import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")

# ================================================================================
# CARGA Y CONFIGURACIÓN INICIAL
# ================================================================================

# Cargar datos
df = pd.read_csv('./Data/intermediate/tani_preprocessed_final_v2.csv')

print("=== FASE I: COMPRENSIÓN DEL PROBLEMA Y ESTRUCTURA DE DATOS ===")
print(f"\nDimensiones del dataset: {df.shape}")
print(f"Período de datos: {df['Fecha'].min()} a {df['Fecha'].max()}")

# ================================================================================
# 1. ANÁLISIS CRÍTICO: NIVEL PACIENTE vs REGISTRO
# ================================================================================

print("\n" + "="*80)
print("1. ANÁLISIS NIVEL PACIENTE vs REGISTRO (CLAVE PARA ENTENDER DESBALANCE)")
print("="*80)

# Estadísticas básicas de estructura
total_registros = len(df)
pacientes_unicos = df['N_HC'].nunique()
promedio_visitas = total_registros / pacientes_unicos

print(f"📊 ESTRUCTURA LONGITUDINAL:")
print(f"   • Total registros (visitas): {total_registros:,}")
print(f"   • Pacientes únicos: {pacientes_unicos:,}")
print(f"   • Promedio visitas por paciente: {promedio_visitas:.2f}")

# Distribución de número de visitas por paciente
visitas_por_paciente = df.groupby('N_HC').size()
print(f"\n📈 DISTRIBUCIÓN DE VISITAS POR PACIENTE:")
print(f"   • Min: {visitas_por_paciente.min()}")
print(f"   • Q25: {visitas_por_paciente.quantile(0.25)}")
print(f"   • Mediana: {visitas_por_paciente.median()}")
print(f"   • Q75: {visitas_por_paciente.quantile(0.75)}")
print(f"   • Max: {visitas_por_paciente.max()}")

# ================================================================================
# 2. ANÁLISIS CRÍTICO DEL DESBALANCE - LA CLAVE DEL PROBLEMA
# ================================================================================

print("\n" + "="*80)
print("2. ANÁLISIS DEL DESBALANCE REAL (PACIENTE vs REGISTRO)")
print("="*80)

# Desbalance a nivel de registro (engañoso por estructura longitudinal)
registros_con_deficit = df[df['flg_alguna'] == 1].shape[0]
print(f"🔍 A NIVEL DE REGISTRO:")
print(f"   • Registros con déficit: {registros_con_deficit:,} ({registros_con_deficit/total_registros*100:.2f}%)")
print(f"   • Registros sin déficit: {total_registros - registros_con_deficit:,} ({(1-registros_con_deficit/total_registros)*100:.2f}%)")

# ANÁLISIS CLAVE: Desbalance a nivel de PACIENTE
# Un paciente con déficit puede tener múltiples registros positivos
pacientes_con_deficit = df.groupby('N_HC')['flg_alguna'].max()  # 1 si alguna vez tuvo déficit
pacientes_deficit_unicos = pacientes_con_deficit.sum()

print(f"\n🎯 A NIVEL DE PACIENTE (ANÁLISIS REAL):")
print(f"   • Pacientes con déficit alguna vez: {pacientes_deficit_unicos:,} ({pacientes_deficit_unicos/pacientes_unicos*100:.2f}%)")
print(f"   • Pacientes nunca con déficit: {pacientes_unicos - pacientes_deficit_unicos:,} ({(1-pacientes_deficit_unicos/pacientes_unicos)*100:.2f}%)")

# Análisis de progresión del déficit en pacientes afectados
deficit_progression = df[df['flg_alguna'] == 1].groupby('N_HC').agg({
    'edad_meses': ['min', 'max', 'count'],
    'primer_alguna': 'first',
    'ultimo_control': 'first'
}).round(2)

deficit_progression.columns = ['edad_primer_deficit', 'edad_ultimo_deficit', 'registros_con_deficit', 
                              'control_primer_deficit', 'ultimo_control']

print(f"\n📊 PROGRESIÓN DEL DÉFICIT EN PACIENTES AFECTADOS:")
print(f"   • Edad promedio primer déficit: {deficit_progression['edad_primer_deficit'].mean():.1f} meses")
print(f"   • Rango edad primer déficit: {deficit_progression['edad_primer_deficit'].min():.1f} - {deficit_progression['edad_primer_deficit'].max():.1f} meses")
print(f"   • Promedio registros con déficit por paciente: {deficit_progression['registros_con_deficit'].mean():.1f}")

# ================================================================================
# 3. ANÁLISIS DE CALIDAD DE DATOS Y MISSING VALUES
# ================================================================================

print("\n" + "="*80)
print("3. CALIDAD DE DATOS Y PATRONES DE MISSING VALUES")
print("="*80)

# Missing values por variable
missing_analysis = pd.DataFrame({
    'Variable': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Pct': (df.isnull().sum() / len(df) * 100).round(2),
    'Unique_Values': [df[col].nunique() for col in df.columns]
})

# Variables con alto porcentaje de missing
high_missing = missing_analysis[missing_analysis['Missing_Pct'] > 20].sort_values('Missing_Pct', ascending=False)

print(f"📋 VARIABLES CON >20% MISSING VALUES:")
if not high_missing.empty:
    for _, row in high_missing.iterrows():
        print(f"   • {row['Variable']}: {row['Missing_Pct']}% missing ({row['Missing_Count']:,} registros)")
else:
    print("   ✓ Ninguna variable tiene >20% missing values")

# Variables categóricas - análisis detallado
categorical_vars = df.select_dtypes(include=['object']).columns.tolist()
print(f"\n📊 VARIABLES CATEGÓRICAS IDENTIFICADAS ({len(categorical_vars)}):")
for var in categorical_vars:
    unique_count = df[var].nunique()
    top_category = df[var].value_counts().index[0] if unique_count > 0 else "N/A"
    top_pct = df[var].value_counts().iloc[0] / df[var].count() * 100 if unique_count > 0 else 0
    print(f"   • {var}: {unique_count} categorías únicas, top: '{top_category}' ({top_pct:.1f}%)")

# ================================================================================
# 4. JUSTIFICACIÓN ESTADÍSTICA DE LA SEGMENTACIÓN PROPUESTA
# ================================================================================

print("\n" + "="*80)
print("FASE II: JUSTIFICACIÓN DE SEGMENTACIÓN POBLACIONAL")
print("="*80)

# Aplicar segmentación propuesta
segmento_propuesto = df[(df['cant_controles_primer_alguna'] >= 6) & (df['ultimo_control'] >= 19)]

print(f"🎯 SEGMENTACIÓN PROPUESTA: cant_controles_primer_alguna >= 6 & ultimo_control >= 19")
print(f"   • Registros en segmento: {len(segmento_propuesto):,} ({len(segmento_propuesto)/len(df)*100:.1f}% del total)")
print(f"   • Pacientes únicos en segmento: {segmento_propuesto['N_HC'].nunique():,}")

# Análisis de prevalencia en segmento vs población general
prevalencia_general_registro = df['flg_alguna'].mean() * 100
prevalencia_segmento_registro = segmento_propuesto['flg_alguna'].mean() * 100

prevalencia_general_paciente = (df.groupby('N_HC')['flg_alguna'].max()).mean() * 100
prevalencia_segmento_paciente = (segmento_propuesto.groupby('N_HC')['flg_alguna'].max()).mean() * 100

print(f"\n📊 COMPARACIÓN DE PREVALENCIAS:")
print(f"   NIVEL REGISTRO:")
print(f"   • Población general: {prevalencia_general_registro:.2f}%")
print(f"   • Segmento propuesto: {prevalencia_segmento_registro:.2f}%")
print(f"   • Ratio: {prevalencia_segmento_registro/prevalencia_general_registro:.2f}x")

print(f"\n   NIVEL PACIENTE:")
print(f"   • Población general: {prevalencia_general_paciente:.2f}%")
print(f"   • Segmento propuesto: {prevalencia_segmento_paciente:.2f}%")
print(f"   • Ratio: {prevalencia_segmento_paciente/prevalencia_general_paciente:.2f}x")

# ================================================================================
# 5. ANÁLISIS DE MADUREZ DEL SEGUIMIENTO
# ================================================================================

print(f"\n📈 ANÁLISIS DE MADUREZ DEL SEGUIMIENTO:")

# Distribución de último control
ultimo_control_stats = df['ultimo_control'].describe()
print(f"   • Último control - Mediana: {ultimo_control_stats['50%']}, Q75: {ultimo_control_stats['75%']}")

# Capacidad de detección por número de controles
deteccion_por_controles = df.groupby('cant_controles_primer_alguna').agg({
    'flg_alguna': ['count', 'sum', 'mean'],
    'N_HC': 'nunique'
}).round(4)

deteccion_por_controles.columns = ['total_registros', 'casos_deficit', 'tasa_deteccion', 'pacientes_unicos']

print(f"\n📊 CAPACIDAD DE DETECCIÓN POR NÚMERO DE CONTROLES (Top 10):")
top_controles = deteccion_por_controles.nlargest(10, 'total_registros')
for idx, row in top_controles.iterrows():
    print(f"   • {idx} controles: {row['casos_deficit']} casos en {row['total_registros']} registros ({row['tasa_deteccion']*100:.3f}%)")

# ================================================================================
# PREPARACIÓN PARA VISUALIZACIONES
# ================================================================================

print(f"\n🎨 PREPARANDO DATOS PARA VISUALIZACIONES...")

# Dataset filtrado con la segmentación propuesta para análisis posterior
df_segmento = segmento_propuesto.copy()

print(f"✅ RESUMEN EJECUTIVO:")
print(f"   • El desbalance real a nivel PACIENTE ({prevalencia_general_paciente:.2f}%) es MÁS ALTO que a nivel registro ({prevalencia_general_registro:.2f}%)")
print(f"   • La segmentación propuesta concentra {prevalencia_segmento_paciente:.2f}% de prevalencia vs {prevalencia_general_paciente:.2f}% general")
print(f"   • {len(df_segmento):,} registros y {df_segmento['N_HC'].nunique():,} pacientes únicos para análisis detallado")

# ================================================================================
# FUNCIONES PARA VISUALIZACIONES (Ejecutar en celdas separadas)
# ================================================================================

def plot_patient_vs_record_analysis():
    """Gráfico comparativo del desbalance paciente vs registro"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Distribución de visitas por paciente
    visitas_por_paciente = df.groupby('N_HC').size()
    ax1.hist(visitas_por_paciente, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_title('Distribución de Visitas por Paciente')
    ax1.set_xlabel('Número de Visitas')
    ax1.set_ylabel('Frecuencia')
    ax1.axvline(visitas_por_paciente.median(), color='red', linestyle='--', label=f'Mediana: {visitas_por_paciente.median():.1f}')
    ax1.legend()
    
    # 2. Comparación desbalance nivel registro vs paciente
    levels = ['Registro', 'Paciente']
    deficit_pcts = [prevalencia_general_registro, prevalencia_general_paciente]
    
    bars = ax2.bar(levels, deficit_pcts, color=['skyblue', 'coral'], alpha=0.8)
    ax2.set_title('Prevalencia de Déficit: Registro vs Paciente')
    ax2.set_ylabel('Porcentaje con Déficit (%)')
    
    # Añadir valores en las barras
    for bar, pct in zip(bars, deficit_pcts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{pct:.3f}%', ha='center', va='bottom')
    
    # 3. Evolución temporal de casos por edad
    edad_deficit = df[df['flg_alguna'] == 1]['edad_meses'].dropna()
    ax3.hist(edad_deficit, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax3.set_title('Distribución de Edad al Detectar Déficit')
    ax3.set_xlabel('Edad (meses)')
    ax3.set_ylabel('Frecuencia')
    ax3.axvline(edad_deficit.median(), color='darkred', linestyle='--', 
               label=f'Mediana: {edad_deficit.median():.1f} meses')
    ax3.legend()
    
    # 4. Segmentación propuesta vs población general
    segments = ['Población\nGeneral', 'Segmento\nPropuesto']
    prevalencias = [prevalencia_general_paciente, prevalencia_segmento_paciente]
    
    bars = ax4.bar(segments, prevalencias, color=['lightblue', 'orange'], alpha=0.8)
    ax4.set_title('Prevalencia por Segmentación (Nivel Paciente)')
    ax4.set_ylabel('Porcentaje con Déficit (%)')
    
    for bar, prev in zip(bars, prevalencias):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{prev:.3f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def analyze_categorical_variables():
    """Análisis detallado de variables categóricas"""
    
    # Variables categóricas clave
    cat_vars_key = ['Sexo', 'Diag_Nacimiento', 'Dx_Nutricional', 'Lactancia', 'T/E_cat', 'P/E_cat', 'P/T_cat']
    
    # Filtrar variables que existen en el dataset
    cat_vars_available = [var for var in cat_vars_key if var in df.columns]
    
    n_vars = len(cat_vars_available)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    for i, var in enumerate(cat_vars_available):
        # Análisis bivariado con el target
        crosstab = pd.crosstab(df[var], df['flg_alguna'], normalize='index') * 100
        
        crosstab.plot(kind='bar', ax=axes[i], color=['lightblue', 'coral'])
        axes[i].set_title(f'{var} vs Déficit de Desarrollo')
        axes[i].set_ylabel('Porcentaje')
        axes[i].legend(['Sin Déficit', 'Con Déficit'])
        axes[i].tick_params(axis='x', rotation=45)
        
    # Ocultar subplots vacíos
    for i in range(len(cat_vars_available), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return cat_vars_available

# ================================================================================
# INSTRUCCIONES PARA EJECUTAR EL ANÁLISIS COMPLETO
# ================================================================================
plot_patient_vs_record_analysis()
analyze_categorical_variables()
print("\n" + "="*80)
print("🚀 INSTRUCCIONES PARA EJECUTAR EL ANÁLISIS COMPLETO")
print("="*80)
print("1. Ejecuta este código base para obtener las estadísticas fundamentales")
print("2. Luego ejecuta: plot_patient_vs_record_analysis()")
print("3. Después ejecuta: analyze_categorical_variables()")
print("4. Continúa con análisis bivariado específico del target")
print("="*80)