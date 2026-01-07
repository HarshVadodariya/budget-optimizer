// Spending Analysis Charts by Occupation and Category

(function() {
  'use strict';

  // Get theme colors
  const getThemeColors = () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    return {
      background: isDark ? '#1e293b' : '#ffffff',
      text: isDark ? '#f1f5f9' : '#0f172a',
      grid: isDark ? '#334155' : '#e2e8f0',
      border: isDark ? '#334155' : '#e2e8f0',
    };
  };

  // Chart.js default configuration
  Chart.defaults.color = getThemeColors().text;
  Chart.defaults.borderColor = getThemeColors().border;
  Chart.defaults.backgroundColor = getThemeColors().background;

  // Color palette for charts
  const chartColors = [
    'rgba(13, 148, 136, 0.8)',   // Teal
    'rgba(99, 102, 241, 0.8)',   // Indigo
    'rgba(16, 185, 129, 0.8)',   // Emerald
    'rgba(245, 158, 11, 0.8)',   // Amber
    'rgba(239, 68, 68, 0.8)',    // Red
    'rgba(139, 92, 246, 0.8)',   // Purple
    'rgba(236, 72, 153, 0.8)',   // Pink
    'rgba(59, 130, 246, 0.8)',   // Blue
  ];

  const chartColorsLight = [
    'rgba(13, 148, 136, 0.3)',
    'rgba(99, 102, 241, 0.3)',
    'rgba(16, 185, 129, 0.3)',
    'rgba(245, 158, 11, 0.3)',
    'rgba(239, 68, 68, 0.3)',
    'rgba(139, 92, 246, 0.3)',
    'rgba(236, 72, 153, 0.3)',
    'rgba(59, 130, 246, 0.3)',
  ];

  let occupationCategoryChart = null;
  let occupationTotalChart = null;
  let categoryComparisonChart = null;

  // Fetch analytics data and render charts
  async function loadCharts() {
    try {
      const response = await fetch('/analytics');
      if (!response.ok) {
        console.warn('Analytics data not available');
        return;
      }

      const data = await response.json();
      renderCharts(data);
    } catch (error) {
      console.error('Error loading analytics:', error);
    }
  }

  function renderCharts(data) {
    const colors = getThemeColors();
    const { occupation_spending, occupation_totals, category_totals, occupations, categories } = data;

    // Chart 1: Stacked Bar Chart - Spending by Occupation & Category
    const ctx1 = document.getElementById('occupationCategoryChart');
    if (ctx1) {
      const categoryLabels = categories.filter(cat => 
        occupation_spending[occupations[0]] && occupation_spending[occupations[0]][cat] !== undefined
      );

      const datasets = occupations.map((occ, idx) => {
        const values = categoryLabels.map(cat => 
          occupation_spending[occ] && occupation_spending[occ][cat] ? occupation_spending[occ][cat] : 0
        );
        return {
          label: occ,
          data: values,
          backgroundColor: chartColors[idx % chartColors.length],
          borderColor: chartColors[idx % chartColors.length].replace('0.8', '1'),
          borderWidth: 2,
        };
      });

      if (occupationCategoryChart) {
        occupationCategoryChart.destroy();
      }

      occupationCategoryChart = new Chart(ctx1, {
        type: 'bar',
        data: {
          labels: categoryLabels,
          datasets: datasets,
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            title: {
              display: false,
            },
            legend: {
              display: true,
              position: 'top',
            },
            tooltip: {
              mode: 'index',
              intersect: false,
              callbacks: {
                label: function(context) {
                  return context.dataset.label + ': ₹' + context.parsed.y.toFixed(2);
                }
              }
            }
          },
          scales: {
            x: {
              stacked: true,
              grid: {
                color: colors.grid,
              },
            },
            y: {
              stacked: true,
              beginAtZero: true,
              grid: {
                color: colors.grid,
              },
              ticks: {
                callback: function(value) {
                  return '₹' + value.toFixed(0);
                }
              }
            }
          }
        }
      });
    }

    // Chart 2: Bar Chart - Total Spending by Occupation
    const ctx2 = document.getElementById('occupationTotalChart');
    if (ctx2) {
      const totals = occupations.map(occ => occupation_totals[occ] || 0);

      if (occupationTotalChart) {
        occupationTotalChart.destroy();
      }

      occupationTotalChart = new Chart(ctx2, {
        type: 'bar',
        data: {
          labels: occupations,
          datasets: [{
            label: 'Total Monthly Spending',
            data: totals,
            backgroundColor: chartColors.map(c => c.replace('0.8', '0.6')),
            borderColor: chartColors,
            borderWidth: 2,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: {
              display: false,
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  return 'Total: ₹' + context.parsed.y.toFixed(2);
                }
              }
            }
          },
          scales: {
            x: {
              grid: {
                color: colors.grid,
              },
            },
            y: {
              beginAtZero: true,
              grid: {
                color: colors.grid,
              },
              ticks: {
                callback: function(value) {
                  return '₹' + value.toFixed(0);
                }
              }
            }
          }
        }
      });
    }

    // Chart 3: Horizontal Bar Chart - Category Spending Comparison
    const ctx3 = document.getElementById('categoryComparisonChart');
    if (ctx3) {
      const categoryLabels = Object.keys(category_totals);
      const categoryValues = Object.values(category_totals);

      if (categoryComparisonChart) {
        categoryComparisonChart.destroy();
      }

      categoryComparisonChart = new Chart(ctx3, {
        type: 'bar',
        data: {
          labels: categoryLabels,
          datasets: [{
            label: 'Average Spending',
            data: categoryValues,
            backgroundColor: chartColors.slice(0, categoryLabels.length),
            borderColor: chartColors.slice(0, categoryLabels.length).map(c => c.replace('0.8', '1')),
            borderWidth: 2,
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: {
              display: false,
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  return 'Average: ₹' + context.parsed.x.toFixed(2);
                }
              }
            }
          },
          scales: {
            x: {
              beginAtZero: true,
              grid: {
                color: colors.grid,
              },
              ticks: {
                callback: function(value) {
                  return '₹' + value.toFixed(0);
                }
              }
            },
            y: {
              grid: {
                color: colors.grid,
              },
            }
          }
        }
      });
    }
  }

  // Update charts when theme changes
  function updateChartsTheme() {
    const colors = getThemeColors();
    Chart.defaults.color = colors.text;
    Chart.defaults.borderColor = colors.border;
    
    if (occupationCategoryChart) {
      occupationCategoryChart.options.scales.x.grid.color = colors.grid;
      occupationCategoryChart.options.scales.y.grid.color = colors.grid;
      occupationCategoryChart.update('none');
    }
    if (occupationTotalChart) {
      occupationTotalChart.options.scales.x.grid.color = colors.grid;
      occupationTotalChart.options.scales.y.grid.color = colors.grid;
      occupationTotalChart.update('none');
    }
    if (categoryComparisonChart) {
      categoryComparisonChart.options.scales.x.grid.color = colors.grid;
      categoryComparisonChart.options.scales.y.grid.color = colors.grid;
      categoryComparisonChart.update('none');
    }
  }

  // Initialize on page load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadCharts);
  } else {
    loadCharts();
  }

  // Listen for theme changes
  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      setTimeout(updateChartsTheme, 100);
    });
  }

  // Expose for external use
  window.updateChartsTheme = updateChartsTheme;
})();

