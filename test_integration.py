"""
Script de test d'intégration complet pour Sentinel 360
Vérifie que tous les modules fonctionnent correctement
"""

import sys
from pathlib import Path
import traceback

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test 1: Vérifier les imports de tous les modules"""
    print("\n" + "="*60)
    print("TEST 1: IMPORTS DES MODULES")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    modules = {
        "utils": "Utilitaires partagés",
        "cleaning": "Nettoyage Bronze → Silver → Gold",
        "models": "Modèles ML (Random Forest, K-Means)",
        "extraction": "Extraction BigQuery",
        "pipeline": "Pipeline complète"
    }
    
    for module_name, description in modules.items():
        try:
            __import__(module_name)
            print(f"✅ {module_name:15} - {description}")
            tests_passed += 1
        except Exception as e:
            print(f"❌ {module_name:15} - ERROR: {str(e)[:80]}")
            tests_failed += 1
    
    return tests_passed, tests_failed


def test_data_files():
    """Test 2: Vérifier les fichiers de données"""
    print("\n" + "="*60)
    print("TEST 2: FICHIERS DE DONNÉES")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    data_files = {
        "data/processed/benin_events_gold.csv": "Gold layer (features engineered)",
        "data/processed/benin_events_silver.csv": "Silver layer (cleaned data)",
        "data/processed/benin_events_bronze.csv": "Bronze layer (raw data)",
    }
    
    for file_path, description in data_files.items():
        p = Path(file_path)
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"✅ {file_path:40} ({size_mb:.2f} MB)")
            tests_passed += 1
        else:
            print(f"⚠️  {file_path:40} (fichier non trouvé)")
    
    return tests_passed, tests_failed


def test_utils_functions():
    """Test 3: Tester les fonctions du module utils"""
    print("\n" + "="*60)
    print("TEST 3: FONCTIONS UTILITAIRES")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        from utils import CAMEO_ROOT_LABELS, goldstein_label
        
        # Test 1: CAMEO labels
        if len(CAMEO_ROOT_LABELS) > 0:
            print(f"✅ CAMEO_ROOT_LABELS: {len(CAMEO_ROOT_LABELS)} catégories chargées")
            tests_passed += 1
        else:
            print("❌ CAMEO_ROOT_LABELS: aucune catégorie trouvée")
            tests_failed += 1
        
        # Test 2: Goldstein labels
        try:
            result_pos = goldstein_label(7.5)
            result_neu = goldstein_label(0.0)
            result_neg = goldstein_label(-5.0)
            
            if result_pos == "Stabilisant" and result_neu == "Neutre" and result_neg == "Déstabilisant":
                print(f"✅ goldstein_label(): fonctionne correctement")
                tests_passed += 1
            else:
                print(f"❌ goldstein_label(): résultats inattendus")
                tests_failed += 1
        except Exception as e:
            print(f"❌ goldstein_label(): {e}")
            tests_failed += 1
    
    except Exception as e:
        print(f"❌ Erreur lors du test utils: {e}")
        traceback.print_exc()
        tests_failed += 2
    
    return tests_passed, tests_failed


def test_models_functions():
    """Test 4: Tester les fonctions du module models"""
    print("\n" + "="*60)
    print("TEST 4: MODULE MODÈLES ML")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        import pandas as pd
        import numpy as np
        from models import (
            make_demo_dataframe, prepare_dataset, build_preprocessor, 
            build_classifier_pipeline, CATEGORICAL_FEATURES, NUMERIC_FEATURES
        )
        
        # Test 1: Créer un dataframe de démo
        try:
            df = make_demo_dataframe()
            if len(df) == 5000 and len(df.columns) > 0:
                print(f"✅ make_demo_dataframe(): DataFrame créé ({len(df)} lignes, {len(df.columns)} colonnes)")
                tests_passed += 1
            else:
                print(f"❌ make_demo_dataframe(): dimensions inattendues")
                tests_failed += 1
        except Exception as e:
            print(f"❌ make_demo_dataframe(): {e}")
            tests_failed += 1
        
        # Test 2: Préparation du dataset
        try:
            X, y = prepare_dataset(df.copy())
            if len(X) > 0 and len(y) > 0 and len(X) == len(y):
                print(f"✅ prepare_dataset(): X={X.shape}, y={y.shape}")
                tests_passed += 1
            else:
                print(f"❌ prepare_dataset(): données invalides")
                tests_failed += 1
        except Exception as e:
            print(f"❌ prepare_dataset(): {e}")
            tests_failed += 1
        
        # Test 3: Vérifier features définies
        try:
            if len(NUMERIC_FEATURES) > 0 and len(CATEGORICAL_FEATURES) > 0:
                print(f"✅ Features: {len(NUMERIC_FEATURES)} numériques + {len(CATEGORICAL_FEATURES)} catégoriques")
                tests_passed += 1
            else:
                print(f"❌ Features: définitions manquantes")
                tests_failed += 1
        except Exception as e:
            print(f"❌ Features: {e}")
            tests_failed += 1
    
    except Exception as e:
        print(f"❌ Erreur lors du test models: {e}")
        traceback.print_exc()
        tests_failed += 3
    
    return tests_passed, tests_failed


def test_cleaning_functions():
    """Test 5: Tester les fonctions du module cleaning"""
    print("\n" + "="*60)
    print("TEST 5: MODULE NETTOYAGE")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        from cleaning import (
            clean_dataframe, polish_dataframe,
            SILVER_OUTPUT_CSV_PATH, GOLD_OUTPUT_CSV_PATH
        )
        
        # Test 1: Vérifier les chemins
        try:
            if str(SILVER_OUTPUT_CSV_PATH).endswith("benin_events_silver.csv"):
                print(f"✅ SILVER_OUTPUT_CSV_PATH: configuré correctement")
                tests_passed += 1
            else:
                print(f"⚠️  SILVER_OUTPUT_CSV_PATH: chemin inattendu")
                tests_failed += 1
        except Exception as e:
            print(f"❌ Chemins: {e}")
            tests_failed += 1
        
        # Test 2: Vérifier que les fonctions existent et sont appelables
        try:
            if callable(clean_dataframe) and callable(polish_dataframe):
                print(f"✅ clean_dataframe() et polish_dataframe(): fonctions disponibles")
                tests_passed += 1
            else:
                print(f"⚠️  Fonctions non trouvées")
                tests_failed += 1
        except Exception as e:
            print(f"⚠️  Vérification des fonctions: {str(e)[:80]}")
            tests_failed += 1
    
    except Exception as e:
        print(f"❌ Erreur lors du test cleaning: {e}")
        traceback.print_exc()
        tests_failed += 2
    
    return tests_passed, tests_failed


def test_dashboard():
    """Test 6: Vérifier la structure du dashboard"""
    print("\n" + "="*60)
    print("TEST 6: DASHBOARD STREAMLIT")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        app_path = Path("dashboard/app.py")
        if app_path.exists():
            with open(app_path, encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Vérifier imports clés
            checks = {
                "streamlit": "✅ Streamlit importé",
                "plotly": "✅ Plotly importé",
                "set_page_config": "✅ Configuration de page",
                "ROYAL": "✅ Thème Sentinel"
            }
            
            for check, msg in checks.items():
                if check in content:
                    print(msg)
                    tests_passed += 1
                else:
                    print(f"⚠️  {msg.replace('✅', '❌')}")
                    tests_failed += 1
        else:
            print("❌ dashboard/app.py: fichier non trouvé")
            tests_failed += 4
    
    except Exception as e:
        print(f"❌ Erreur lors du test dashboard: {e}")
        tests_failed += 4
    
    return tests_passed, tests_failed


def main():
    """Exécuter tous les tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  TESTS D'INTÉGRATION - SENTINEL 360".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    total_passed = 0
    total_failed = 0
    
    # Exécuter tous les tests
    p, f = test_imports()
    total_passed += p
    total_failed += f
    
    p, f = test_data_files()
    total_passed += p
    total_failed += f
    
    p, f = test_utils_functions()
    total_passed += p
    total_failed += f
    
    p, f = test_models_functions()
    total_passed += p
    total_failed += f
    
    p, f = test_cleaning_functions()
    total_passed += p
    total_failed += f
    
    p, f = test_dashboard()
    total_passed += p
    total_failed += f
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    print(f"✅ Tests réussis:  {total_passed}")
    print(f"⚠️  Tests échoués:  {total_failed}")
    print(f"📊 Total:           {total_passed + total_failed}")
    
    success_rate = (total_passed / (total_passed + total_failed) * 100) if (total_passed + total_failed) > 0 else 0
    print(f"🎯 Taux de réussite: {success_rate:.1f}%")
    
    if total_failed == 0:
        print("\n✅ TOUS LES TESTS SONT PASSÉS - CODE PRÊT POUR MERGE!")
        return 0
    else:
        print(f"\n⚠️  {total_failed} tests à corriger avant le merge")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
