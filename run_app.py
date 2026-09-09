# run_app.py - ENHANCED Application Launcher with Universal Symbols & Portfolio
import os
import sys
import subprocess
import webbrowser
import time
import socket
from datetime import datetime

def check_port_available(port):
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if check_port_available(port):
            return port
    return None

def install_requirements():
    """Install required packages"""
    print("📦 Checking and installing required packages...")
    
    requirements = [
        'flask',
        'flask-login',
        'yfinance',
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'plotly',
        'seaborn'
    ]
    
    for package in requirements:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Installing...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")

def check_system_health():
    """Check system health and dependencies"""
    print("\n🔍 System Health Check:")
    print("=" * 40)
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python Version: {python_version}")
    
    # Check available memory (approximate)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Available Memory: {memory.available // (1024**3)} GB / {memory.total // (1024**3)} GB")
    except ImportError:
        print("Memory: psutil not installed (optional)")
    
    # Check disk space
    try:
        disk = psutil.disk_usage('.')
        print(f"Disk Space: {disk.free // (1024**3)} GB free")
    except:
        print("Disk Space: Check skipped")
    
    print("=" * 40)

def display_welcome():
    """Display welcome message and feature overview"""
    print("""
    🚀 STOCK PREDICTOR PRO - ENHANCED LAUNCHER
    ==========================================
    
    🌟 NEW FEATURES:
    • 🌍 Universal Symbol Support (Any stock worldwide)
    • 💼 Real Portfolio Management
    • 🎯 Best Buy/Sell Date Detection
    • 📊 Real-time Market Data
    • 🤖 Multiple AI Models (LSTM, GRU, Ensemble)
    • 📈 Advanced Technical Analysis
    • 💰 Portfolio Performance Tracking
    
    🎯 QUICK START:
    • Choose Flask App for full web interface
    • Choose Streamlit for data science workflow
    • All apps support the same enhanced features
    """)

def create_directories():
    """Create necessary directories"""
    directories = [
        'templates',
        'static',
        'reports',
        'data_cache',
        'portfolios',
        'models',
        'charts'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def run_flask_app():
    """Run the Flask application with enhanced features"""
    print("\n🔥 Starting Flask Application...")
    
    # Find available port
    port = find_available_port(5000)
    if not port:
        print("❌ No available ports found. Trying port 5001...")
        port = find_available_port(5001)
        if not port:
            print("❌ No ports available. Please close other applications and try again.")
            return
    
    print(f"🌐 Application will be available at: http://localhost:{port}")
    print("⏳ Starting server...")
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(3)
        webbrowser.open(f"http://localhost:{port}")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # ``wsgi.py`` reads ``PORT`` from the application settings.  FLASK_RUN_PORT
    # only applies to the Flask CLI and previously meant the launcher could
    # report one URL while the application still tried to bind to port 5000.
    os.environ['PORT'] = str(port)
    
    try:
        # Run the maintained application factory rather than the legacy,
        # monolithic app.py entry point.  subprocess also avoids shell parsing
        # and correctly propagates the selected port to wsgi.py.
        if os.path.exists('wsgi.py'):
            print("🚀 Launching Stock Predictor Pro...")
            subprocess.call([sys.executable, 'wsgi.py'])
        else:
            print("❌ wsgi.py not found. Please ensure the file exists.")
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")

def run_streamlit_app(app_file):
    """Run Streamlit application"""
    print(f"\n🔥 Starting {app_file}...")
    
    # Find available port
    port = find_available_port(8501)
    if not port:
        port = find_available_port(8502)
    
    if port:
        print(f"🌐 Application will be available at: http://localhost:{port}")
        
        # Open browser after delay
        def open_browser():
            time.sleep(5)
            webbrowser.open(f"http://localhost:{port}")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            os.system(f'streamlit run {app_file} --server.port {port}')
        except KeyboardInterrupt:
            print("\n🛑 Application stopped by user")
        except Exception as e:
            print(f"❌ Error starting Streamlit app: {e}")
    else:
        print("❌ No available ports for Streamlit")

def run_all_apps():
    """Run all applications in different ports"""
    print("\n🚀 Starting All Applications...")
    
    apps = [
        ('Flask App', 'app.py', 5000),
        ('Streamlit Basic', 'predict.py', 8501),
        ('Streamlit Advanced', 'app_streamlit.py', 8502)
    ]
    
    processes = []
    
    for app_name, app_file, default_port in apps:
        if os.path.exists(app_file):
            port = find_available_port(default_port)
            if port:
                print(f"🌐 {app_name} at http://localhost:{port}")
                
                if app_name == 'Flask App':
                    # Flask app in separate process
                    import subprocess
                    env = os.environ.copy()
                    env['FLASK_RUN_PORT'] = str(port)
                    process = subprocess.Popen([sys.executable, app_file], env=env)
                else:
                    # Streamlit app
                    process = subprocess.Popen([
                        sys.executable, '-m', 'streamlit', 'run', app_file, 
                        '--server.port', str(port), '--server.headless', 'true'
                    ])
                
                processes.append(process)
                time.sleep(2)  # Stagger startup
            else:
                print(f"❌ No port available for {app_name}")
        else:
            print(f"❌ {app_file} not found")
    
    if processes:
        print(f"\n✅ {len(processes)} applications started!")
        print("📋 Open the following URLs in your browser:")
        for app_name, app_file, default_port in apps:
            port = find_available_port(default_port - 1) or default_port
            print(f"   • {app_name}: http://localhost:{port}")
        
        print("\n🛑 Press Ctrl+C to stop all applications")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping all applications...")
            for process in processes:
                process.terminate()
            for process in processes:
                process.wait()
            print("✅ All applications stopped")

def test_features():
    """Test enhanced features"""
    print("\n🧪 Testing Enhanced Features...")
    
    tests = [
        ("Universal Symbol Validation", "test_symbols"),
        ("Real-time Data Fetching", "test_realtime"),
        ("Portfolio System", "test_portfolio"),
        ("AI Model Loading", "test_models")
    ]
    
    for test_name, test_type in tests:
        try:
            if test_type == "test_symbols":
                # Test universal symbol validation
                from model_utils import validate_stock_symbol
                symbols = ['AAPL', 'TSLA', 'SHOP.TO', 'HSBA.L', 'RELIANCE.NS']
                valid_count = 0
                for symbol in symbols:
                    result = validate_stock_symbol(symbol)
                    if result.get('valid'):
                        valid_count += 1
                print(f"✅ {test_name}: {valid_count}/{len(symbols)} symbols valid")
                
            elif test_type == "test_realtime":
                # Test real-time data
                from model_utils import get_current_real_price
                price = get_current_real_price('AAPL')
                if price > 0:
                    print(f"✅ {test_name}: Real-time data working (AAPL: ${price:.2f})")
                else:
                    print(f"❌ {test_name}: Real-time data failed")
                    
            elif test_type == "test_portfolio":
                # Test portfolio system
                from model_utils import PortfolioManager
                portfolio = PortfolioManager("test_user")
                summary = portfolio.get_portfolio_summary()
                if summary:
                    print(f"✅ {test_name}: Portfolio system working")
                else:
                    print(f"❌ {test_name}: Portfolio system failed")
                    
            elif test_type == "test_models":
                # Test AI models
                try:
                    from predictor_core import AdvancedStockPredictor
                    print(f"✅ {test_name}: AI models loaded successfully")
                except ImportError as e:
                    print(f"⚠️ {test_name}: AI models partially available - {e}")
                    
        except Exception as e:
            print(f"❌ {test_name}: Failed - {e}")

def main():
    """Enhanced main function"""
    display_welcome()
    
    # System checks
    check_system_health()
    
    # Install requirements
    install_requirements()
    
    # Create directories
    create_directories()
    
    # Test features
    test_features()
    
    while True:
        print("\n" + "="*50)
        print("🎯 APPLICATION LAUNCHER MENU")
        print("="*50)
        print("1. 🚀 Flask Web App (Recommended)")
        print("   • Full web interface with all features")
        print("   • Universal symbol support")
        print("   • Portfolio management")
        print("   • Real-time data")
        
        print("\n2. 📊 Streamlit Basic App")
        print("   • Simple prediction interface")
        print("   • Quick analysis")
        print("   • Clean, modern UI")
        
        print("\n3. 🤖 Streamlit Advanced App")
        print("   • Comprehensive analysis")
        print("   • Multiple AI models")
        print("   • Advanced visualizations")
        
        print("\n4. 🌟 Run ALL Applications")
        print("   • Start Flask + both Streamlit apps")
        print("   • Different ports for each")
        
        print("\n5. 🧪 Test System & Features")
        print("   • Run diagnostic tests")
        print("   • Verify all features work")
        
        print("\n0. ❌ Exit")
        print("="*50)
        
        choice = input("\nChoose option (0-5): ").strip()
        
        if choice == "1":
            run_flask_app()
            break
            
        elif choice == "2":
            if os.path.exists('predict.py'):
                run_streamlit_app('predict.py')
                break
            else:
                print("❌ predict.py not found")
                
        elif choice == "3":
            if os.path.exists('app_streamlit.py'):
                run_streamlit_app('app_streamlit.py')
                break
            else:
                print("❌ app_streamlit.py not found")
                
        elif choice == "4":
            run_all_apps()
            break
            
        elif choice == "5":
            test_features()
            input("\nPress Enter to continue...")
            
        elif choice == "0":
            print("👋 Thank you for using Stock Predictor Pro!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Application launcher stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your installation and try again.")
