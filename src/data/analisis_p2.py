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
# DEFINICIÓN DE ESTRATEGIA: ¿NIVEL PACIENTE O REGISTRO?
# ================================================================================

print("\n" + "="*80)
print("🚨 DECISIÓN CRÍTICA: NIVEL DE ANÁLISIS")
print("="*80)
print("OPCIÓN A: NIVEL PACIENTE (Recomendado para predicción)")
print("   • Un registro por paciente (última visita o agregado)")
print("   • Target: ¿Alguna vez tuvo déficit?")
print("   • Prevalencia real: 0.94%")
print("   • Mejor para modelo predictivo")
print("")
print("OPCIÓN B: NIVEL REGISTRO (Para análisis temporal)")
print("   • Todos los registros (454,901)")
print("   • Target: ¿Tiene déficit en esta visita específica?")
print("   • Prevalencia aparente: 0.10%")
print("   • Útil para análisis de progresión temporal")
print("="*80)

# ================================================================================
# FUNCIÓN PARA CREAR DATASET A NIVEL PACIENTE
# ================================================================================

def create_patient_level_dataset(segmented=True):
    """
    Crea dataset a nivel paciente agregando información por N_HC
    
    segmented: Si True, aplica la segmentación propuesta
    """
    
    # Aplicar segmentación si se solicita
    if segmented:
        df_work = df[(df['cant_controles_primer_alguna'] >= 6) & (df['ultimo_control'] >= 19)].copy()
        print(f"📊 Aplicando segmentación: {len(df_work):,} registros de {df_work['N_HC'].nunique():,} pacientes")
    else:
        df_work = df.copy()
        print(f"📊 Dataset completo: {len(df_work):,} registros de {df_work['N_HC'].nunique():,} pacientes")
    
    # Agregar a nivel paciente
    patient_agg = df_work.groupby('N_HC').agg({
        # TARGET: ¿Alguna vez tuvo déficit?
        'flg_alguna': 'max',
        'flg_cognitivo': 'max',
        'flg_lenguaje': 'max', 
        'flg_motora_fina': 'max',
        'flg_motora_gruesa': 'max',
        'flg_social': 'max',
        
        # Características del paciente (última visita o más frecuente)
        'Sexo': 'last',
        'Diag_Nacimiento': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[-1],
        'edad_meses': 'max',  # Edad en última visita
        
        # Variables antropométricas (última medición)
        'Peso': 'last',
        'Talla': 'last', 
        'CabPC': 'last',
        'T/E_cat': 'last',
        'P/E_cat': 'last',
        'P/T_cat': 'last',
        
        # Variables nutricionales (más frecuente o última)
        'Dx_Nutricional': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[-1],
        'Lactancia': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None,
        'ACA': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None,
        
        # Variables de seguimiento
        'ultimo_control': 'max',
        'cantidad_controles': 'last',
        'cant_controles_primer_alguna': 'last',
        'primer_alguna': 'first',
        
        # Fechas
        'Fecha': ['first', 'last'],
        
        # Contadores
        'N_HC': 'count'  # Número de visitas
    }).round(2)
    
    # Aplanar columnas multinivel
    patient_agg.columns = ['_'.join(col).strip() if col[1] != '' else col[0] for col in patient_agg.columns.values]
    patient_agg.rename(columns={'N_HC_count': 'num_visitas'}, inplace=True)
    
    # Limpiar nombres de columnas
    patient_agg.columns = [col.replace('_<lambda>', '_mode').replace('_last', '').replace('_max', '') 
                          for col in patient_agg.columns]
    
    # Estadísticas del dataset creado
    target_prevalence = patient_agg['flg_alguna'].mean() * 100
    print(f"✅ Dataset nivel paciente creado:")
    print(f"   • Pacientes: {len(patient_agg):,}")
    print(f"   • Prevalencia target: {target_prevalence:.2f}%")
    print(f"   • Promedio visitas: {patient_agg['num_visitas'].mean():.1f}")
    
    return patient_agg

# ================================================================================
# VISUALIZACIONES CORREGIDAS A NIVEL PACIENTE
# ================================================================================

def plot_patient_level_analysis(df_patients):
    """Análisis visual a nivel paciente (SIN desbalance extremo)"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Distribución del target a nivel paciente
    target_counts = df_patients['flg_alguna'].value_counts()
    target_pct = df_patients['flg_alguna'].value_counts(normalize=True) * 100
    
    bars1 = ax1.bar(['Sin Déficit', 'Con Déficit'], target_counts.values, 
                   color=['lightblue', 'coral'], alpha=0.8)
    ax1.set_title('Distribución del Target (Nivel Paciente)')
    ax1.set_ylabel('Número de Pacientes')
    
    for bar, count, pct in zip(bars1, target_counts.values, target_pct.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{count:,}\n({pct:.2f}%)', ha='center', va='bottom')
    
    # 2. Distribución de edad al último control
    ax2.hist(df_patients['edad_meses'].dropna(), bins=30, alpha=0.7, 
             color='skyblue', edgecolor='black')
    ax2.set_title('Distribución de Edad (Última Visita)')
    ax2.set_xlabel('Edad (meses)')
    ax2.set_ylabel('Frecuencia')
    ax2.axvline(df_patients['edad_meses'].median(), color='red', linestyle='--',
               label=f'Mediana: {df_patients["edad_meses"].median():.1f} meses')
    ax2.legend()
    
    # 3. Número de visitas por paciente
    ax3.hist(df_patients['num_visitas'], bins=30, alpha=0.7, 
             color='lightgreen', edgecolor='black')
    ax3.set_title('Distribución de Número de Visitas por Paciente')
    ax3.set_xlabel('Número de Visitas')
    ax3.set_ylabel('Frecuencia')
    ax3.axvline(df_patients['num_visitas'].median(), color='red', linestyle='--',
               label=f'Mediana: {df_patients["num_visitas"].median():.1f}')
    ax3.legend()
    
    # 4. Comparación por sexo
    if 'Sexo' in df_patients.columns:
        sexo_target = pd.crosstab(df_patients['Sexo'], df_patients['flg_alguna'], 
                                 normalize='index') * 100
        sexo_target.plot(kind='bar', ax=ax4, color=['lightblue', 'coral'])
        ax4.set_title('Prevalencia de Déficit por Sexo')
        ax4.set_ylabel('Porcentaje')
        ax4.legend(['Sin Déficit', 'Con Déficit'])
        ax4.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()

def plot_categorical_analysis_patient_level(df_patients):
    """Análisis bivariado de variables categóricas a nivel paciente"""
    
    categorical_vars = ['Diag_Nacimiento', 'Dx_Nutricional', 'T/E_cat', 'P/E_cat', 'P/T_cat', 'Lactancia']
    
    # Filtrar variables disponibles
    available_vars = [var for var in categorical_vars if var in df_patients.columns]
    
    n_vars = len(available_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    print("📊 ANÁLISIS BIVARIADO A NIVEL PACIENTE:")
    print("="*60)
    
    for i, var in enumerate(available_vars):
        # Calcular crosstab
        ct = pd.crosstab(df_patients[var], df_patients['flg_alguna'], normalize='index') * 100
        
        if ct.shape[1] > 1:  # Si hay casos de déficit
            ct.plot(kind='bar', ax=axes[i], color=['lightblue', 'coral'])
            axes[i].set_title(f'{var} vs Déficit')
            axes[i].set_ylabel('Porcentaje')
            axes[i].legend(['Sin Déficit', 'Con Déficit'])
            axes[i].tick_params(axis='x', rotation=45)
            
            # Mostrar estadísticas de riesgo
            print(f"\n{var}:")
            if 1 in ct.columns:
                risk_by_category = ct[1].sort_values(ascending=False)
                for category, risk in risk_by_category.head(3).items():
                    if risk > 0:
                        print(f"   • {category}: {risk:.2f}% riesgo")
    
    # Ocultar subplots vacíos
    for i in range(len(available_vars), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# ================================================================================
# EJECUCIÓN RECOMENDADA
# ================================================================================

print("\n🎯 RECOMENDACIÓN: Ejecutar análisis A NIVEL PACIENTE")
print("="*60)
print("1. df_patients = create_patient_level_dataset(segmented=True)")
print("2. plot_patient_level_analysis(df_patients)")
print("3. plot_categorical_analysis_patient_level(df_patients)")
print("="*60)
df_patients = create_patient_level_dataset(segmented=True)
df_patients_segmented = create_patient_level_dataset(segmented=True)
plot_patient_level_analysis(df_patients)
plot_categorical_analysis_patient_level(df_patients)
# ================================================================================
# FUNCIONES CORREGIDAS QUE USAN EL DATASET A NIVEL PACIENTE
# ================================================================================

def analyze_development_flags_patient_level(df_patients):
    """Análisis detallado de flags a NIVEL PACIENTE (corregido)"""
    
    development_flags = ['flg_cognitivo', 'flg_lenguaje', 'flg_motora_fina', 
                        'flg_motora_gruesa', 'flg_social', 'flg_alguna']
    
    print("\n" + "="*80)
    print("ANÁLISIS DE FLAGS A NIVEL PACIENTE (CORREGIDO)")
    print("="*80)
    
    # Estadísticas a nivel paciente únicamente
    flag_stats = {}
    available_flags = [flag for flag in development_flags if flag in df_patients.columns]
    
    for flag in available_flags:
        total_patients = df_patients[flag].notna().sum()
        deficit_patients = df_patients[flag].sum()
        prevalence = (deficit_patients/total_patients*100) if total_patients > 0 else 0
        
        flag_stats[flag] = {
            'total_patients': total_patients,
            'deficit_patients': deficit_patients,
            'prevalence_pct': prevalence
        }
    
    # Mostrar estadísticas
    print(f"{'Flag':<20} {'Pacientes Total':<15} {'Con Déficit':<12} {'Prevalencia(%)':<15}")
    print("-" * 65)
    
    for flag, stats in flag_stats.items():
        print(f"{flag:<20} {stats['total_patients']:<15.0f} {stats['deficit_patients']:<12.0f} "
              f"{stats['prevalence_pct']:<15.2f}")
    
    # Visualización
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Prevalencia por flag a nivel paciente
    flags_names = list(flag_stats.keys())
    patient_prevalences = [flag_stats[flag]['prevalence_pct'] for flag in flags_names]
    
    bars1 = ax1.bar(range(len(flags_names)), patient_prevalences, color='coral', alpha=0.8)
    ax1.set_title('Prevalencia de Déficit por Área (Nivel Paciente)')
    ax1.set_ylabel('Porcentaje de Pacientes (%)')
    ax1.set_xticks(range(len(flags_names)))
    ax1.set_xticklabels([f.replace('flg_', '').replace('_', '\n') for f in flags_names], rotation=0)
    
    # Añadir valores en las barras
    for bar, prev in zip(bars1, patient_prevalences):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{prev:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Comorbilidad - Número de déficits por paciente
    available_dev_flags = [f for f in available_flags if f != 'flg_alguna']
    if available_dev_flags:
        df_patients['total_deficits'] = df_patients[available_dev_flags].sum(axis=1)
        comorbidity = df_patients['total_deficits'].value_counts().sort_index()
        
        bars2 = ax2.bar(comorbidity.index, comorbidity.values, color='skyblue', alpha=0.8)
        ax2.set_title('Distribución de Número de Déficits por Paciente')
        ax2.set_xlabel('Número de Áreas con Déficit')
        ax2.set_ylabel('Número de Pacientes')
        
        for bar, count in zip(bars2, comorbidity.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    str(count), ha='center', va='bottom')
    
    # 3. Distribución de edad en pacientes con y sin déficit
    if 'edad_meses' in df_patients.columns:
        edad_sin_deficit = df_patients[df_patients['flg_alguna'] == 0]['edad_meses'].dropna()
        edad_con_deficit = df_patients[df_patients['flg_alguna'] == 1]['edad_meses'].dropna()
        
        ax3.hist(edad_sin_deficit, alpha=0.6, label='Sin Déficit', bins=20, color='lightblue')
        ax3.hist(edad_con_deficit, alpha=0.6, label='Con Déficit', bins=20, color='coral')
        ax3.set_title('Distribución de Edad por Presencia de Déficit')
        ax3.set_xlabel('Edad (meses)')
        ax3.set_ylabel('Frecuencia')
        ax3.legend()
    
    # 4. Número de visitas vs déficit
    if 'num_visitas' in df_patients.columns:
        visitas_sin_deficit = df_patients[df_patients['flg_alguna'] == 0]['num_visitas'].dropna()
        visitas_con_deficit = df_patients[df_patients['flg_alguna'] == 1]['num_visitas'].dropna()
        
        ax4.hist(visitas_sin_deficit, alpha=0.6, label='Sin Déficit', bins=20, color='lightgreen')
        ax4.hist(visitas_con_deficit, alpha=0.6, label='Con Déficit', bins=20, color='red')
        ax4.set_title('Distribución de Número de Visitas por Déficit')
        ax4.set_xlabel('Número de Visitas')
        ax4.set_ylabel('Frecuencia')
        ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    return flag_stats
analyze_development_flags_patient_level(df_patients)

def analyze_categorical_patient_level(df_patients):
    """Análisis de variables categóricas A NIVEL PACIENTE (corregido)"""
    
    print("\n" + "="*80)
    print("ANÁLISIS CATEGÓRICO A NIVEL PACIENTE (CORREGIDO)")
    print("="*80)
    
    # Variables categóricas disponibles
    categorical_vars = ['Sexo', 'Diag_Nacimiento', 'Dx_Nutricional', 'T/E_cat', 'P/E_cat', 'P/T_cat', 'Lactancia']
    available_vars = [var for var in categorical_vars if var in df_patients.columns]
    
    n_vars = len(available_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    print("📊 ANÁLISIS BIVARIADO A NIVEL PACIENTE:")
    print("="*60)
    
    risk_summary = {}
    
    for i, var in enumerate(available_vars):
        # Filtrar valores no nulos
        df_var = df_patients[df_patients[var].notna()]
        
        if len(df_var) > 0:
            # Calcular crosstab
            ct = pd.crosstab(df_var[var], df_var['flg_alguna'], normalize='index') * 100
            
            if ct.shape[1] > 1:  # Si hay casos de déficit
                ct.plot(kind='bar', ax=axes[i], color=['lightblue', 'coral'])
                axes[i].set_title(f'{var} vs Déficit (N={len(df_var):,} pacientes)')
                axes[i].set_ylabel('Porcentaje')
                axes[i].legend(['Sin Déficit', 'Con Déficit'])
                axes[i].tick_params(axis='x', rotation=45)
                
                # Estadísticas de riesgo
                print(f"\n{var} (N={len(df_var):,} pacientes):")
                risk_by_category = ct[1].sort_values(ascending=False)
                risk_summary[var] = risk_by_category
                
                for category, risk in risk_by_category.head(5).items():
                    count = pd.crosstab(df_var[var], df_var['flg_alguna']).loc[category].sum()
                    print(f"   • {category}: {risk:.2f}% riesgo (N={count})")
            else:
                axes[i].text(0.5, 0.5, f'Sin casos de déficit\nen {var}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{var} - Sin déficit detectado')
    
    # Ocultar subplots vacíos
    for i in range(len(available_vars), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return risk_summary
analyze_categorical_patient_level(df_patients)

def analyze_nutritional_patient_level(df_patients):
    """Análisis nutricional y antropométrico A NIVEL PACIENTE (corregido)"""
    
    print("\n" + "="*80)
    print("ANÁLISIS NUTRICIONAL A NIVEL PACIENTE (CORREGIDO)")
    print("="*80)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Peso vs déficit
    if 'Peso' in df_patients.columns:
        peso_sin = df_patients[df_patients['flg_alguna'] == 0]['Peso'].dropna()
        peso_con = df_patients[df_patients['flg_alguna'] == 1]['Peso'].dropna()
        
        ax1.hist(peso_sin, alpha=0.6, label='Sin Déficit', bins=20, color='lightblue')
        ax1.hist(peso_con, alpha=0.6, label='Con Déficit', bins=20, color='coral')
        ax1.set_title('Distribución de Peso por Déficit')
        ax1.set_xlabel('Peso (kg)')
        ax1.set_ylabel('Frecuencia')
        ax1.legend()
    
    # 2. Talla vs déficit
    if 'Talla' in df_patients.columns:
        talla_sin = df_patients[df_patients['flg_alguna'] == 0]['Talla'].dropna()
        talla_con = df_patients[df_patients['flg_alguna'] == 1]['Talla'].dropna()
        
        ax2.hist(talla_sin, alpha=0.6, label='Sin Déficit', bins=20, color='lightgreen')
        ax2.hist(talla_con, alpha=0.6, label='Con Déficit', bins=20, color='red')
        ax2.set_title('Distribución de Talla por Déficit')
        ax2.set_xlabel('Talla (cm)')
        ax2.set_ylabel('Frecuencia')
        ax2.legend()
    
    # 3. P/T_cat vs déficit
    if 'P/T_cat' in df_patients.columns:
        ct_pt = pd.crosstab(df_patients['P/T_cat'], df_patients['flg_alguna'], normalize='index') * 100
        if ct_pt.shape[1] > 1:
            ct_pt.plot(kind='bar', ax=ax3, color=['lightblue', 'coral'])
            ax3.set_title('Peso/Talla vs Déficit')
            ax3.set_ylabel('Porcentaje')
            ax3.legend(['Sin Déficit', 'Con Déficit'])
            ax3.tick_params(axis='x', rotation=45)
    
    # 4. T/E_cat vs déficit
    if 'T/E_cat' in df_patients.columns:
        ct_te = pd.crosstab(df_patients['T/E_cat'], df_patients['flg_alguna'], normalize='index') * 100
        if ct_te.shape[1] > 1:
            ct_te.plot(kind='bar', ax=ax4, color=['lightgreen', 'red'])
            ax4.set_title('Talla/Edad vs Déficit')
            ax4.set_ylabel('Porcentaje')
            ax4.legend(['Sin Déficit', 'Con Déficit'])
            ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
analyze_nutritional_patient_level(df_patients)
# ================================================================================
# INSTRUCCIONES CORREGIDAS PARA EJECUTAR
# ================================================================================

print("\n" + "="*80)
print("🎯 INSTRUCCIONES CORREGIDAS (AHORA SÍ NIVEL PACIENTE)")
print("="*80)
print("# PASO 1: Crear dataset nivel paciente")
print("df_patients = create_patient_level_dataset(segmented=True)")
print("")
print("# PASO 2: Análisis visual principal (CORREGIDO)")
print("plot_patient_level_analysis(df_patients)")
print("")
print("# PASO 3: Análisis de flags (CORREGIDO)")
print("development_stats = analyze_development_flags_patient_level(df_patients)")
print("")
print("# PASO 4: Análisis categórico (CORREGIDO)")
print("risk_summary = analyze_categorical_patient_level(df_patients)")
print("")
print("# PASO 5: Análisis nutricional (CORREGIDO)")
print("analyze_nutritional_patient_level(df_patients)")
print("="*80)

# ================================================================================
# INSTRUCCIONES FINALES
# ================================================================================

print("\n" + "="*80)
print("🎯 HALLAZGOS CLAVE Y PRÓXIMOS PASOS")
print("="*80)
print("✅ CONFIRMAMOS: Desbalance real a nivel paciente (0.94%) vs registro (0.10%)")
print("✅ SEGMENTACIÓN: Concentra casos (1.77% vs 0.94%) - JUSTIFICADA")
print("✅ TEMPORALIDAD: Déficit se detecta principalmente 15-35 meses")
print("✅ READY: Dataset filtrado con 269,158 registros para modelado")
print("\n🚀 PRÓXIMO PASO: Análisis de correlaciones y preparación features para ML")
print("="*80)