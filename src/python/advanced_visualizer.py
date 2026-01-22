# src/python/advanced_visualizer.py
#!/usr/bin/env python3
"""
Visualizador Avanzado de Constant Hunter - Versión Mejorada
Análisis completo de clustering, distribución espacial y estadísticas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import argparse

# Configurar matplotlib para mejor calidad
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#2b2b2b'
plt.rcParams['axes.facecolor'] = '#2b2b2b'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['grid.color'] = '#555555'
plt.rcParams['grid.alpha'] = 0.3

class AdvancedVisualizer:
    """Visualizador avanzado con todas las funcionalidades"""

    def __init__(self, results_dir=None):
        self.results_dir = Path(results_dir) if results_dir else self._find_latest_results()
        self.data = {}
        self.stats = {}
        self.constants_info = {
            'c': {'name': 'Velocidad de la luz', 'length': 9, 'value': '299792458'},
            'h': {'name': 'Constante de Planck', 'length': 9, 'value': '662607015'},
            'hbar': {'name': 'Constante de Planck reducida', 'length': 10, 'value': '1054571817'},
            'mu0': {'name': 'Permeabilidad magnética', 'length': 12, 'value': '125663706127'},
            'Z0': {'name': 'Impedancia del vacío', 'length': 12, 'value': '376730313412'},
            'epsilon0': {'name': 'Permitividad del vacío', 'length': 11, 'value': '88541878188'},
            'k': {'name': 'Constante de Boltzmann', 'length': 7, 'value': '1380649'},
            'G': {'name': 'Constante gravitacional', 'length': 6, 'value': '667430'},
            'sigma': {'name': 'Constante de Stefan-Boltzmann', 'length': 10, 'value': '5670374419'}
        }

    def _find_latest_results(self):
        """Encuentra el directorio de resultados más reciente"""
        possible_dirs = [
            Path.cwd() / "results",
            Path(__file__).parent.parent / "results",
            Path("/home/padmin/Descargas/Constant Hunter v.1/results"),
            Path.home() / "Descargas" / "Constant Hunter v.1" / "results"
        ]

        for dir_path in possible_dirs:
            if dir_path.exists():
                result_dirs = list(dir_path.glob('results_*'))
                if result_dirs:
                    latest = max(result_dirs, key=os.path.getmtime)
                    return latest

        raise FileNotFoundError("No se encontraron directorios de resultados")

    def load_results(self):
        """Carga y procesa todos los resultados"""
        print(f"📁 Cargando resultados de: {self.results_dir}")

        for file_path in self.results_dir.glob('*_*.txt'):
            if 'SUMMARY' in file_path.name:
                continue

            const_name = file_path.name.split('_')[0]
            positions = []

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                in_positions = False
                for line in lines:
                    line = line.strip()

                    if 'POSICIONES ENCONTRADAS' in line:
                        in_positions = True
                        continue

                    if in_positions and line and line[0].isdigit():
                        try:
                            pos = int(line.split()[0])
                            positions.append(pos)
                        except:
                            continue

                if positions:
                    self.data[const_name] = np.array(sorted(positions))
                    print(f"  ✅ {const_name}: {len(positions)} posiciones")

            except Exception as e:
                print(f"  ❌ Error en {file_path.name}: {e}")

        return self.data

    def analyze_complete(self, pi_size=1000000002):
        """Análisis estadístico completo"""
        print("\n📊 Realizando análisis estadístico completo...")

        for const_name, positions in self.data.items():
            if len(positions) < 2:
                continue

            # Estadísticas básicas
            stats_dict = {
                'count': len(positions),
                'density_mb': len(positions) / (pi_size / 1e6),
                'min_pos': int(positions.min()),
                'max_pos': int(positions.max()),
                'mean_pos': float(positions.mean()),
                'median_pos': float(np.median(positions)),
                'std_pos': float(positions.std()),
                'q1': float(np.percentile(positions, 25)),
                'q3': float(np.percentile(positions, 75)),
                'iqr': float(np.percentile(positions, 75) - np.percentile(positions, 25))
            }

            # Análisis de gaps
            if len(positions) > 1:
                gaps = np.diff(positions)

                stats_dict.update({
                    'mean_gap': float(gaps.mean()),
                    'median_gap': float(np.median(gaps)),
                    'std_gap': float(gaps.std()),
                    'min_gap': int(gaps.min()),
                    'max_gap': int(gaps.max()),
                    'cv_gap': float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else 0,
                    'q1_gap': float(np.percentile(gaps, 25)),
                    'q3_gap': float(np.percentile(gaps, 75)),
                    'iqr_gap': float(np.percentile(gaps, 75) - np.percentile(gaps, 25))
                })

                # Test de aleatoriedad
                try:
                    ks_test = stats.kstest(positions / pi_size, 'uniform', args=(0, 1))
                    stats_dict.update({
                        'ks_statistic': float(ks_test.statistic),
                        'ks_pvalue': float(ks_test.pvalue),
                        'is_uniform': ks_test.pvalue > 0.05
                    })
                except:
                    pass

                # Análisis de autocorrelación
                if len(positions) > 10:
                    try:
                        lags = min(20, len(positions) - 1)
                        autocorr = np.correlate(positions - positions.mean(),
                                               positions - positions.mean(),
                                               mode='full')
                        autocorr = autocorr[len(autocorr)//2:][:lags]
                        autocorr = autocorr / autocorr[0]

                        stats_dict['autocorrelation_lag1'] = float(autocorr[1])
                        stats_dict['autocorrelation_lag5'] = float(autocorr[5]) if len(autocorr) > 5 else 0
                    except:
                        pass

            # Análisis de clustering
            cluster_stats = self._analyze_clustering_detailed(positions)
            stats_dict.update(cluster_stats)

            self.stats[const_name] = stats_dict

        return self.stats

    def _analyze_clustering_detailed(self, positions, thresholds=None):
        """Análisis detallado de clustering"""
        if thresholds is None:
            thresholds = [100, 500, 1000, 5000, 10000]

        cluster_results = {}

        for threshold in thresholds:
            clusters = []
            current_cluster = [positions[0]]

            for i in range(1, len(positions)):
                if positions[i] - positions[i-1] <= threshold:
                    current_cluster.append(positions[i])
                else:
                    if len(current_cluster) > 1:
                        clusters.append(np.array(current_cluster))
                    current_cluster = [positions[i]]

            if len(current_cluster) > 1:
                clusters.append(np.array(current_cluster))

            key = f'clusters_{threshold}'
            cluster_results[key] = {
                'num_clusters': len(clusters),
                'total_in_clusters': sum(len(c) for c in clusters),
                'mean_cluster_size': float(np.mean([len(c) for c in clusters])) if clusters else 0,
                'largest_cluster': max([len(c) for c in clusters]) if clusters else 0
            }

        return cluster_results

    def create_comprehensive_report(self):
        """Crea reporte JSON completo"""
        report = {
            "analysis_date": datetime.now().isoformat(),
            "results_directory": str(self.results_dir),
            "pi_file_size": 1000000002,
            "total_constants_searched": 9,
            "constants_found": len([c for c, pos in self.data.items() if len(pos) > 0]),
            "total_occurrences": sum(len(pos) for pos in self.data.values()),
            "search_statistics": self.stats,
            "constants_info": self.constants_info,
            "data_summary": {
                const_name: {
                    "count": len(positions),
                    "positions_sample": positions[:20].tolist() if len(positions) > 20 else positions.tolist()
                }
                for const_name, positions in self.data.items()
            }
        }

        return report

    def create_all_visualizations(self, output_dir="advanced_visualizations"):
        """Crea todas las visualizaciones"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n🎨 Generando visualizaciones en: {output_path}")

        # 1. Distribución de densidad comparativa
        self._create_density_comparison_plot(output_path, timestamp)

        # 2. Análisis de gaps
        self._create_gap_analysis_plots(output_path, timestamp)

        # 3. Gráficos de clustering
        self._create_clustering_plots(output_path, timestamp)

        # 4. Análisis estadístico resumen
        self._create_statistical_summary_plot(output_path, timestamp)

        # 5. Reporte JSON
        self._create_json_report(output_path, timestamp)

        return output_path

    def _create_density_comparison_plot(self, output_path, timestamp):
        """Crea gráfico de densidad comparativa"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        pi_size = 1000000002

        # 1. Densidad KDE
        ax = axes[0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.data)))

        for idx, (const_name, positions) in enumerate(self.data.items()):
            if len(positions) > 1:
                normalized = positions / pi_size

                try:
                    kde = stats.gaussian_kde(normalized, bw_method=0.05)
                    x_vals = np.linspace(0, 1, 1000)
                    y_vals = kde(x_vals)

                    ax.plot(x_vals * 100, y_vals,
                           label=f"{const_name} (n={len(positions)})",
                           color=colors[idx], linewidth=1.5, alpha=0.8)
                except:
                    # Histograma como fallback
                    hist, bins = np.histogram(normalized, bins=50, density=True)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    ax.plot(bin_centers * 100, hist,
                           label=f"{const_name} (n={len(positions)})",
                           color=colors[idx], linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Posición en Pi (%)")
        ax.set_ylabel("Densidad de Probabilidad")
        ax.set_title("Distribución de Densidad - Comparativa")
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

        # 2. Posiciones acumulativas
        ax = axes[1]
        for idx, (const_name, positions) in enumerate(self.data.items()):
            if len(positions) > 0:
                sorted_pos = np.sort(positions)
                cumulative = np.arange(1, len(sorted_pos) + 1) / len(sorted_pos)

                ax.plot(sorted_pos / 1e6, cumulative,
                       label=const_name, color=colors[idx], linewidth=1.5)

        ax.set_xlabel("Posición (Millones de dígitos)")
        ax.set_ylabel("Proporción Acumulativa")
        ax.set_title("Función de Distribución Acumulativa")
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)

        # 3. Boxplot de posiciones
        ax = axes[2]
        data_to_plot = []
        labels = []

        for const_name, positions in self.data.items():
            if len(positions) > 0:
                data_to_plot.append(positions / 1e6)  # Convertir a millones
                labels.append(f"{const_name}\n(n={len(positions)})")

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

            # Colorear los boxplots
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel("Posición (Millones de dígitos)")
            ax.set_title("Distribución de Posiciones - Boxplot")
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # 4. Heatmap de correlación
        ax = axes[3]
        if len(self.data) > 1:
            # Crear matriz de posiciones normalizadas
            max_len = max(len(pos) for pos in self.data.values())
            norm_matrix = []
            const_names = []

            for const_name, positions in self.data.items():
                if len(positions) > 10:
                    normalized = positions / pi_size
                    # Interpolar para tener la misma longitud
                    if len(normalized) < max_len:
                        x_old = np.linspace(0, 1, len(normalized))
                        x_new = np.linspace(0, 1, max_len)
                        normalized = np.interp(x_new, x_old, normalized)

                    norm_matrix.append(normalized[:max_len])
                    const_names.append(const_name)

            if len(norm_matrix) > 1:
                corr_matrix = np.corrcoef(norm_matrix)

                im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_xticks(range(len(const_names)))
                ax.set_yticks(range(len(const_names)))
                ax.set_xticklabels(const_names, rotation=45, ha='right')
                ax.set_yticklabels(const_names)
                ax.set_title("Matriz de Correlación entre Constantes")

                # Añadir valores
                for i in range(len(const_names)):
                    for j in range(len(const_names)):
                        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                     ha="center", va="center",
                                     color="white" if abs(corr_matrix[i, j]) < 0.5 else "black")

                plt.colorbar(im, ax=ax)

        plt.suptitle("Análisis Avanzado de Distribución de Constantes en Pi",
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / f"comprehensive_analysis_{timestamp}.png",
                   dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

        print("  ✅ Gráfico de análisis completo generado")

    def _create_gap_analysis_plots(self, output_path, timestamp):
        """Crea gráficos de análisis de gaps"""
        for const_name, positions in self.data.items():
            if len(positions) > 10:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                gaps = np.diff(positions)

                # 1. Histograma de gaps
                ax = axes[0, 0]
                ax.hist(gaps, bins=min(50, len(gaps)), edgecolor='white',
                       alpha=0.7, color='#2196F3', density=True)
                ax.axvline(np.mean(gaps), color='red', linestyle='--',
                          linewidth=2, label=f'Media: {np.mean(gaps):.0f}')
                ax.axvline(np.median(gaps), color='orange', linestyle=':',
                          linewidth=2, label=f'Mediana: {np.median(gaps):.0f}')
                ax.set_xlabel("Gap (dígitos)")
                ax.set_ylabel("Densidad")
                ax.set_title(f"Distribución de Gaps - {const_name}")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # 2. Gráfico Q-Q (comparación con exponencial)
                ax = axes[0, 1]
                if len(gaps) > 10:
                    try:
                        stats.probplot(gaps, dist="expon", plot=ax)
                        ax.set_title(f"Gráfico Q-Q vs Distribución Exponencial")
                        ax.grid(True, alpha=0.3)
                    except:
                        ax.text(0.5, 0.5, "No se pudo crear Q-Q plot",
                               ha='center', va='center')

                # 3. Lag plot (autocorrelación)
                ax = axes[1, 0]
                if len(gaps) > 2:
                    ax.scatter(gaps[:-1], gaps[1:], alpha=0.6, s=20)
                    ax.set_xlabel("Gap en posición i")
                    ax.set_ylabel("Gap en posición i+1")
                    ax.set_title(f"Lag Plot - {const_name}")
                    ax.grid(True, alpha=0.3)

                # 4. Análisis acumulativo
                ax = axes[1, 1]
                cumulative_gaps = np.cumsum(gaps)
                ax.plot(range(len(cumulative_gaps)), cumulative_gaps,
                       linewidth=1.5, color='#4CAF50')
                ax.set_xlabel("Índice de gap")
                ax.set_ylabel("Suma acumulativa de gaps")
                ax.set_title(f"Gaps Acumulativos - {const_name}")
                ax.grid(True, alpha=0.3)

                plt.suptitle(f"Análisis Detallado de Gaps: {const_name} ({len(positions)} ocurrencias)",
                           fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                plt.savefig(output_path / f"gap_analysis_{const_name}_{timestamp}.png",
                          dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
                plt.close()

        print("  ✅ Gráficos de análisis de gaps generados")

    def _create_clustering_plots(self, output_path, timestamp):
        """Crea gráficos de clustering detallados"""
        for const_name, positions in self.data.items():
            if len(positions) > 20:
                fig = plt.figure(figsize=(16, 8))

                # Detectar clusters con múltiples thresholds
                thresholds = [100, 500, 1000, 5000]

                for idx, threshold in enumerate(thresholds):
                    ax = plt.subplot(2, 2, idx + 1)

                    clusters = []
                    current_cluster = [positions[0]]

                    for i in range(1, len(positions)):
                        if positions[i] - positions[i-1] <= threshold:
                            current_cluster.append(positions[i])
                        else:
                            if len(current_cluster) > 1:
                                clusters.append(current_cluster)
                            current_cluster = [positions[i]]

                    if len(current_cluster) > 1:
                        clusters.append(current_cluster)

                    # Visualizar clusters
                    for cluster_idx, cluster in enumerate(clusters):
                        y_pos = cluster_idx % 10  # Para distribuir verticalmente
                        ax.scatter(cluster, [y_pos] * len(cluster),
                                  s=30, alpha=0.7, label=f'Cluster {cluster_idx+1}' if cluster_idx < 3 else "")

                        # Conectar puntos del cluster
                        if len(cluster) > 1:
                            ax.plot(cluster, [y_pos] * len(cluster),
                                   '--', alpha=0.5, linewidth=0.5)

                    ax.set_xlabel("Posición (dígitos)")
                    ax.set_ylabel("Cluster ID")
                    ax.set_title(f"Threshold = {threshold} dígitos\nClusters: {len(clusters)}")
                    ax.grid(True, alpha=0.2)

                    if idx == 0 and clusters:
                        ax.legend(fontsize=8)

                plt.suptitle(f"Análisis de Clustering Multi-threshold: {const_name}",
                           fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                plt.savefig(output_path / f"clustering_{const_name}_{timestamp}.png",
                          dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
                plt.close()

        print("  ✅ Gráficos de clustering generados")

    def _create_statistical_summary_plot(self, output_path, timestamp):
        """Crea gráfico resumen estadístico"""
        if not self.stats:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Densidad por constante
        ax = axes[0, 0]
        constants = list(self.stats.keys())
        densities = [self.stats[c]['density_mb'] for c in constants]

        bars = ax.bar(range(len(constants)), densities,
                     color=plt.cm.viridis(np.linspace(0.2, 0.8, len(constants))))
        ax.set_xticks(range(len(constants)))
        ax.set_xticklabels(constants, rotation=45)
        ax.set_ylabel("Densidad (ocurrencias por MB)")
        ax.set_title("Densidad de Ocurrencias")
        ax.grid(True, alpha=0.3, axis='y')

        # Añadir valores en barras
        for bar, density in zip(bars, densities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{density:.3f}', ha='center', va='bottom', fontsize=9)

        # 2. Media de gaps
        ax = axes[0, 1]
        mean_gaps = [self.stats[c].get('mean_gap', 0) for c in constants]

        bars = ax.bar(range(len(constants)), mean_gaps,
                     color=plt.cm.plasma(np.linspace(0.2, 0.8, len(constants))))
        ax.set_xticks(range(len(constants)))
        ax.set_xticklabels(constants, rotation=45)
        ax.set_ylabel("Gap promedio (dígitos)")
        ax.set_title("Distancia Promedio entre Ocurrencias")
        ax.grid(True, alpha=0.3, axis='y')

        # 3. KS p-values (uniformidad)
        ax = axes[1, 0]
        ks_pvalues = []
        labels = []

        for c in constants:
            if 'ks_pvalue' in self.stats[c]:
                ks_pvalues.append(self.stats[c]['ks_pvalue'])
                labels.append(c)

        if ks_pvalues:
            colors = ['#4CAF50' if p > 0.05 else '#F44336' for p in ks_pvalues]
            bars = ax.bar(range(len(labels)), ks_pvalues, color=colors)

            ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5,
                      label='Umbral 0.05')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_ylabel("KS p-value")
            ax.set_title("Test de Uniformidad (Kolmogorov-Smirnov)")
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        # 4. Número de clusters (threshold 1000)
        ax = axes[1, 1]
        cluster_counts = []
        cluster_labels = []

        for c in constants:
            key = 'clusters_1000'
            if key in self.stats[c]:
                cluster_counts.append(self.stats[c][key]['num_clusters'])
                cluster_labels.append(c)

        if cluster_counts:
            ax.bar(range(len(cluster_labels)), cluster_counts,
                  color=plt.cm.Set3(np.linspace(0, 1, len(cluster_labels))))
            ax.set_xticks(range(len(cluster_labels)))
            ax.set_xticklabels(cluster_labels, rotation=45)
            ax.set_ylabel("Número de clusters")
            ax.set_title("Clusters detectados (threshold=1000 dígitos)")
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle("Resumen Estadístico de Análisis de Constantes",
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / f"statistical_summary_{timestamp}.png",
                   dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

        print("  ✅ Gráfico de resumen estadístico generado")

    def _create_json_report(self, output_path, timestamp):
        """Crea reporte JSON detallado"""
        report = self.create_comprehensive_report()

        json_path = output_path / f"detailed_analysis_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # También crear CSV resumen
        csv_path = output_path / f"statistics_summary_{timestamp}.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Constante,Count,Density_MB,Mean_Gap,Std_Gap,Min_Gap,Max_Gap,KS_pvalue,Uniform,Clusters_100\n")

            for const_name, stats in self.stats.items():
                row = [
                    const_name,
                    str(stats.get('count', 0)),
                    f"{stats.get('density_mb', 0):.6f}",
                    f"{stats.get('mean_gap', 0):.0f}",
                    f"{stats.get('std_gap', 0):.0f}",
                    str(stats.get('min_gap', 0)),
                    str(stats.get('max_gap', 0)),
                    f"{stats.get('ks_pvalue', 0):.6f}" if 'ks_pvalue' in stats else "N/A",
                    str(stats.get('is_uniform', False)) if 'is_uniform' in stats else "N/A",
                    str(stats.get('clusters_100', {}).get('num_clusters', 0)) if 'clusters_100' in stats else "0"
                ]
                f.write(','.join(row) + '\n')

        print(f"  ✅ Reportes JSON/CSV generados")
        return json_path, csv_path

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Visualizador Avanzado de Constant Hunter')
    parser.add_argument('--results-dir', type=str, help='Directorio de resultados')
    parser.add_argument('--output-dir', type=str, default='advanced_visualizations',
                       help='Directorio de salida para visualizaciones')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("   VISUALIZADOR AVANZADO - ANÁLISIS DE CONSTANTES EN PI")
    print("="*70)

    try:
        # Crear visualizador
        visualizer = AdvancedVisualizer(args.results_dir)

        # Cargar resultados
        print("\n[1/3] 📂 Cargando resultados...")
        data = visualizer.load_results()

        if not data:
            print("❌ No se encontraron resultados para analizar")
            return

        print(f"   ✓ Cargadas {len(data)} constantes con resultados")

        # Análisis estadístico
        print("\n[2/3] 📊 Realizando análisis estadístico...")
        stats = visualizer.analyze_complete()

        # Mostrar resumen
        print("\n   RESUMEN ESTADÍSTICO:")
        print("   " + "-"*60)
        for const_name, s in stats.items():
            print(f"   📈 {const_name}:")
            print(f"      • Ocurrencias: {s['count']}")
            print(f"      • Densidad: {s['density_mb']:.3f}/MB")
            print(f"      • Gap promedio: {s['mean_gap']:.0f} ± {s['std_gap']:.0f} dígitos")
            if 'ks_pvalue' in s:
                uniform = "SÍ" if s['is_uniform'] else "NO"
                print(f"      • Distribución uniforme: {uniform} (p={s['ks_pvalue']:.4f})")
            if 'clusters_1000' in s:
                print(f"      • Clusters detectados: {s['clusters_1000']['num_clusters']}")
            print()

        # Generar visualizaciones
        print("\n[3/3] 🎨 Generando visualizaciones...")
        output_path = visualizer.create_all_visualizations(args.output_dir)

        # Reporte final
        total_occurrences = sum(len(pos) for pos in data.values())
        constants_with_data = len([c for c, pos in data.items() if len(pos) > 0])

        print("\n" + "="*70)
        print("   ✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"\n   📊 ESTADÍSTICAS FINALES:")
        print(f"      • Constantes analizadas: {len(data)}")
        print(f"      • Constantes con ocurrencias: {constants_with_data}")
        print(f"      • Total de ocurrencias: {total_occurrences}")
        print(f"      • Archivos generados: {len(list(output_path.glob('*')))}")
        print(f"\n   📁 DIRECTORIO DE SALIDA:")
        print(f"      {output_path.absolute()}")
        print(f"\n   📄 ARCHIVOS GENERADOS:")
        for file in sorted(output_path.glob('*')):
            size_kb = file.stat().st_size / 1024
            print(f"      • {file.name} ({size_kb:.1f} KB)")
        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
