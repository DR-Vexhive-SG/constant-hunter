"""
Información de constantes físicas para Constant Hunter
"""

CONSTANTS_INFO = {
    'c': {
        'name': 'Velocidad de la luz en vacío',
        'digits': '299792458',
        'length': 9,
        'description': 'Velocidad de la luz en el vacío (m/s)'
    },
    'h': {
        'name': 'Constante de Planck',
        'digits': '662607015',
        'length': 9,
        'description': 'Constante de Planck (J·s)'
    },
    'hbar': {
        'name': 'Constante de Planck reducida',
        'digits': '1054571817',
        'length': 10,
        'description': 'Constante de Planck reducida (ħ = h/2π)'
    },
    'G': {
        'name': 'Constante gravitacional',
        'digits': '667430',
        'length': 6,
        'description': 'Constante de gravitación universal (m³/kg·s²)'
    },
    'k': {
        'name': 'Constante de Boltzmann',
        'digits': '1380649',
        'length': 7,
        'description': 'Constante de Boltzmann (J/K)'
    },
    'mu0': {
        'name': 'Permeabilidad magnética del vacío',
        'digits': '125663706127',
        'length': 12,
        'description': 'Permeabilidad magnética del vacío (N/A²)'
    },
    'epsilon0': {
        'name': 'Permitividad eléctrica del vacío',
        'digits': '88541878188',
        'length': 11,
        'description': 'Permitividad eléctrica del vacío (F/m)'
    },
    'sigma': {
        'name': 'Constante de Stefan-Boltzmann',
        'digits': '5670374419',
        'length': 10,
        'description': 'Constante de Stefan-Boltzmann (W/m²·K⁴)'
    },
    'Z0': {
        'name': 'Impedancia característica del vacío',
        'digits': '376730313412',
        'length': 12,
        'description': 'Impedancia característica del vacío (Ω)'
    }
}

# Constantes adicionales para búsqueda personalizada
ADDITIONAL_CONSTANTS = {
    'alpha': {
        'name': 'Constante de estructura fina',
        'digits': '729735257',
        'length': 9,
        'description': 'Constante de estructura fina (α ≈ 1/137)'
    },
    'phi': {
        'name': 'Número áureo (φ)',
        'digits': '161803398',
        'length': 9,
        'description': 'Número áureo, proporción áurea'
    },
    'avogadro': {
        'name': 'Número de Avogadro',
        'digits': '602214076',
        'length': 9,
        'description': 'Número de Avogadro (mol⁻¹)'
    }
}

def get_constant_info(const_name):
    """Obtiene información de una constante"""
    if const_name in CONSTANTS_INFO:
        return CONSTANTS_INFO[const_name]
    elif const_name in ADDITIONAL_CONSTANTS:
        return ADDITIONAL_CONSTANTS[const_name]
    else:
        return {
            'name': f'Constante {const_name}',
            'digits': '',
            'length': 0,
            'description': 'Constante personalizada'
        }

def get_all_constants():
    """Obtiene todas las constantes disponibles"""
    all_constants = {}
    all_constants.update(CONSTANTS_INFO)
    all_constants.update(ADDITIONAL_CONSTANTS)
    return all_constants
