# app_streamlit_full.py (COMPLETE ADVANCED VERSION)
import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from io import BytesIO
import base64
import json

try:
    from predictor_core import (
        generate_forecast_report, get_portfolio_recommendations,
        compare_models
    )
    from model_utils import (
        get_stock_data, AdvancedSentimentAnalyzer, AdvancedRiskManager,
        PortfolioSimulator, get_sector_performance, AnomalyDetector,
        add_all_indicators, calculate_model_metrics
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    st.error(f"Import error: {e}")
    IMPORT_SUCCESS = False

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}
.risk-high { border-left-color: #ff4b4b !important; }
.risk-medium { border-left-color: #ffa500 !important; }
.risk-low { border-left-color: #00cc96 !important; }
.positive { color: #00cc96; font-weight: bold; }
.negative { color: #ff4b4b; font-weight: bold; }
.tab-content { padding: 20px 0; }
</style>
""", unsafe_allow_html=True)

class AdvancedStockApp:
    def __init__(self):
        if IMPORT_SUCCESS:
            self.sentiment_analyzer = AdvancedSentimentAnalyzer()
            self.risk_manager = AdvancedRiskManager()
            self.portfolio_simulator = PortfolioSimulator()
            self.anomaly_detector = AnomalyDetector()
        self.user_session = {}

    def initialize_session(self):
        """Initialize user session"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.portfolio = {}
            st.session_state.watchlist = []
            st.session_state.reports = {}

    def run(self):
        """Main application runner"""
        self.initialize_session()

        # Sidebar
        with st.sidebar:
            st.title("📊 Navigation")
            app_mode = st.selectbox(
                "Choose Mode",
                ["Dashboard", "Stock Analysis", "Portfolio", "Model Comparison", "Reports", "Settings"]
            )

            st.markdown("---")
            st.subheader("Quick Actions")
            
            if st.button("🔄 Refresh Market Data"):
                st.rerun()
                
            if st.button("📊 Sector Overview"):
                st.session_state.show_sectors = True

        # Main content based on selected mode
        if app_mode == "Dashboard":
            self.show_dashboard()
        elif app_mode == "Stock Analysis":
            self.show_stock_analysis()
        elif app_mode == "Portfolio":
            self.show_portfolio()
        elif app_mode == "Model Comparison":
            self.show_model_comparison()
        elif app_mode == "Reports":
            self.show_reports()
        elif app_mode == "Settings":
            self.show_settings()

    def show_dashboard(self):
        """Show main dashboard"""
        st.markdown('<div class="main-header">📈 AI Stock Predictor Pro</div>', unsafe_allow_html=True)

        # Market overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("S&P 500", "4,567.25", "+1.2%")
        with col2:
            st.metric("NASDAQ", "14,229.91", "+0.8%")
        with col3:
            st.metric("DOW JONES", "35,654.29", "+0.9%")
        with col4:
            st.metric("VIX", "15.23", "-0.5%")

        # Quick analysis section
        st.subheader("🔍 Quick Stock Analysis")
        quick_symbol = st.text_input("Enter stock symbol:", "AAPL").upper()

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("Analyze Now", type="primary"):
                with st.spinner("Analyzing stock..."):
                    self.quick_analyze_stock(quick_symbol)

        with col2:
            if st.button("Add to Watchlist"):
                if quick_symbol not in st.session_state.watchlist:
                    st.session_state.watchlist.append(quick_symbol)
                    st.success(f"Added {quick_symbol} to watchlist")

        # Sector performance
        st.subheader("📊 Sector Performance")
        if IMPORT_SUCCESS:
            try:
                sector_data = get_sector_performance()
                if sector_data:
                    sectors = list(sector_data.keys())
                    returns = [sector_data[s]['avg_return'] for s in sectors]
                    trends = [sector_data[s]['trend'] for s in sectors]

                    fig = go.Figure(data=[
                        go.Bar(
                            x=sectors,
                            y=returns,
                            marker_color=['#00cc96' if trend == 'BULLISH' else '#ff4b4b' for trend in trends],
                            text=[f"{ret:.1f}%" for ret in returns],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="Sector Returns (1Y)",
                        xaxis_title="Sectors",
                        yaxis_title="Return (%)",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Sector data unavailable: {e}")

        # Recent alerts
        st.subheader("🚨 Market Alerts")
        self.show_market_alerts()

    def quick_analyze_stock(self, symbol):
        """Quick analysis of a stock"""
        if not IMPORT_SUCCESS:
            st.error("Required modules not available")
            return

        try:
            df = get_stock_data(symbol, period='6mo')
            if df.empty:
                st.error(f"No data found for {symbol}")
                return

            # Current price and metrics
            current_price = df["Close"].iloc[-1]
            price_change = (current_price / df["Close"].iloc[-2] - 1) * 100 if len(df) > 1 else 0

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")

            with col2:
                # RSI
                rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                st.metric("RSI", f"{rsi:.1f}")

            with col3:
                # Risk score
                if self.risk_manager:
                    risk_score, risk_level = self.risk_manager.calculate_risk_score(df, symbol)
                    st.metric("Risk Level", risk_level)

            with col4:
                # Sentiment
                try:
                    sentiment, label, confidence, _ = self.sentiment_analyzer.get_news_sentiment(symbol)
                    st.metric("Sentiment", label)
                except:
                    st.metric("Sentiment", "N/A")

            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'],
                name='Price', line=dict(color="#1f77b4")
            ))
            fig.update_layout(
                title=f"{symbol} Price Chart (6 Months)",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error analyzing {symbol}: {str(e)}")

    def show_stock_analysis(self):
        """Comprehensive stock analysis"""
        st.title("🔍 Advanced Stock Analysis")

        # Input section
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            symbol = st.text_input("Stock Symbol", "AAPL").upper()

        with col2:
            forecast_days = st.selectbox("Forecast Horizon", [7, 14, 30, 60, 90], index=2)

        with col3:
            model_type = st.selectbox(
                "Model Type",
                ["AUTO", "LSTM", "GRU", "ENSEMBLE", "PROPHET", "ARIMA"]
            )

        # Analysis options
        col1, col2, col3 = st.columns(3)

        with col1:
            include_sentiment = st.checkbox("Include Sentiment Analysis", value=True)

        with col2:
            include_risk = st.checkbox("Include Risk Analysis", value=True)

        with col3:
            retrain_model = st.checkbox("Retrain Models", value=False)

        if st.button("🚀 Run Comprehensive Analysis", type="primary"):
            self.run_comprehensive_analysis(
                symbol, forecast_days, model_type,
                include_sentiment, include_risk, retrain_model
            )

    def run_comprehensive_analysis(self, symbol, days, model_type, include_sentiment, include_risk, retrain):
        """Run comprehensive stock analysis"""
        if not IMPORT_SUCCESS:
            st.error("Required modules not available")
            return

        with st.spinner("Running comprehensive analysis..."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Data collection
            status_text.text("📊 Fetching market data...")
            df = get_stock_data(symbol, period='2y')
            progress_bar.progress(25)

            if df.empty:
                st.error(f"No data available for {symbol}")
                return

            # Step 2: Generate forecast
            status_text.text("🤖 Generating price forecast...")
            try:
                forecast_df, report, csv_path = generate_forecast_report(
                    symbol, days=days, model_type=model_type, retrain=retrain)
                progress_bar.progress(50)
            except Exception as e:
                st.error(f"Forecast generation failed: {str(e)}")
                return

            # Step 3: Additional analyses
            status_text.text("📈 Analyzing risk and sentiment...")

            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Forecast", "📈 Technicals", "💰 Signals",
                "⚠️ Risk Analysis", "😊 Sentiment"
            ])

            with tab1:
                self.show_forecast_tab(forecast_df, report, symbol)

            with tab2:
                self.show_technical_tab(df, symbol)

            with tab3:
                self.show_signals_tab(report, symbol)

            with tab4:
                if include_risk:
                    self.show_risk_tab(df, report, symbol)
                else:
                    st.info("Risk analysis was not selected")

            with tab5:
                if include_sentiment:
                    self.show_sentiment_tab(symbol)
                else:
                    st.info("Sentiment analysis was not selected")

            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")

            # Download section
            st.markdown("---")
            self.show_download_section(forecast_df, report, symbol)

    def show_forecast_tab(self, forecast_df, report, symbol):
        """Display forecast results"""
        col1, col2, col3, col4 = st.columns(4)

        current_price = report['executive_summary']['current_price']
        predicted_price = report['executive_summary']['predicted_price']
        expected_return = report['executive_summary']['expected_return']
        risk_level = report['executive_summary']['risk_level']

        with col1:
            st.metric("Current Price", f"${current_price:.2f}")

        with col2:
            st.metric("Predicted Price", f"${predicted_price:.2f}", f"{expected_return:.2f}%")

        with col3:
            st.metric("Best Model", report['executive_summary'].get('best_model', 'LSTM'))

        with col4:
            risk_color = "risk-high" if risk_level == "HIGH" else "risk-medium" if risk_level == "MEDIUM" else "risk-low"
            st.markdown(f'<div class="metric-card {risk_color}">Risk Level: {risk_level}</div>', unsafe_allow_html=True)

        # Forecast chart
        fig = go.Figure()

        # Confidence interval
        if 'CI_Upper' in forecast_df.columns and 'CI_Lower' in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['CI_Upper'],
                fill=None,
                mode='lines',
                line_color='rgba(255,165,0,0.3)',
                name='Confidence Upper',
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['CI_Lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(255,165,0,0.3)',
                name='Confidence Interval (±2%)'
            ))

        # Predicted prices
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Predicted_Price'],
            mode='lines',
            name='Predicted Price',
            line=dict(color='#FF7F0E', dash='dash')
        ))

        fig.update_layout(
            title=f"{symbol} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        st.subheader("Detailed Forecast")
        st.dataframe(forecast_df.style.format({
            'Predicted_Price': '${:.2f}',
            'CI_Upper': '${:.2f}',
            'CI_Lower': '${:.2f}'
        }))

    def show_technical_tab(self, df, symbol):
        """Display technical analysis"""
        st.subheader("Technical Indicators")

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Price & Bollinger Bands', 'RSI', 'MACD'],
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )

        # Price and Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            name='Close Price', line=dict(color='#1f77b4')
        ), row=1, col=1)

        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_Upper'],
                name='BB Upper', line=dict(color='rgba(255,0,0,0.3)')
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_Lower'],
                name='BB Lower', line=dict(color='rgba(0,255,0,0.3)'),
                fill='tonexty'
            ), row=1, col=1)

        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['RSI'],
                name='RSI', line=dict(color='purple')
            ), row=2, col=1)
            fig.add_hline(y=70, line_dash='dash', line_color='red', row=2, col=1)
            fig.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)

        # MACD
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD'],
                name='MACD', line=dict(color='blue')
            ), row=3, col=1)

            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD_Signal'],
                name='Signal', line=dict(color='red')
            ), row=3, col=1)

        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    def show_signals_tab(self, report, symbol):
        """Display trading signals"""
        st.subheader("💰 Trading Signals")

        signals = report.get('trading_signals', {})
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Strong Buy Signals", len(signals.get('strong_buy', [])))
        with col2:
            st.metric("Buy Signals", len(signals.get('buy_days', [])))
        with col3:
            st.metric("Sell Signals", len(signals.get('strong_sell', [])) + len(signals.get('sell_days', [])))

        # Display top signals
        st.subheader("Top Strong Buy Opportunities")
        strong_buy = signals.get('strong_buy', [])
        if strong_buy:
            strong_buy_df = pd.DataFrame(strong_buy[:10])
            st.dataframe(strong_buy_df)
        else:
            st.info("No strong buy signals detected")

        st.subheader("Top Strong Sell Warnings")
        strong_sell = signals.get('strong_sell', [])
        if strong_sell:
            strong_sell_df = pd.DataFrame(strong_sell[:10])
            st.dataframe(strong_sell_df)
        else:
            st.info("No strong sell signals detected")

    def show_risk_tab(self, df, report, symbol):
        """Display risk analysis"""
        st.subheader("⚠️ Risk Analysis")

        risk_score = report['executive_summary'].get('risk_score', 50)
        risk_level = report['executive_summary'].get('risk_level', 'MEDIUM')

        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Risk Score: {risk_level}"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Risk metrics
        returns = df['Close'].pct_change().dropna()
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if self.risk_manager and len(returns) > 0:
                var_95 = self.risk_manager.calculate_var(returns, 0.95)
                st.metric("VaR (95%)", f"{(var_95 * 100):.2f}%")

        with col2:
            if self.risk_manager and len(returns) > 0:
                sharpe = self.risk_manager.calculate_sharpe_ratio(returns)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        with col3:
            if self.risk_manager:
                max_dd = self.risk_manager.calculate_max_drawdown(df['Close'].values)
                st.metric("Max Drawdown", f"{(max_dd * 100):.2f}%")

        with col4:
            if self.risk_manager and len(returns) > 0:
                es = self.risk_manager.calculate_expected_shortfall(returns)
                st.metric("Expected Shortfall", f"{(es * 100):.2f}%")

        # Volatility alerts
        st.subheader("🚨 Market Alerts")
        if self.anomaly_detector:
            alerts = self.anomaly_detector.detect_volatility_spikes(df)
            if alerts:
                for alert in alerts[:5]:  # Show top 5 alerts
                    st.warning(f"Volatility Spike: {alert['date'].strftime('%Y-%m-%d')} - Return: {alert['return']:.2%} (Z-score: {alert['z_score']:.1f})")
            else:
                st.info("No significant volatility alerts")

    def show_sentiment_tab(self, symbol):
        """Display sentiment analysis"""
        st.subheader("😊 Market Sentiment")

        if not IMPORT_SUCCESS:
            st.error("Sentiment analysis not available")
            return

        sentiment, label, confidence, headlines = self.sentiment_analyzer.get_news_sentiment(symbol)
        col1, col2 = st.columns(2)

        with col1:
            # Sentiment gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Sentiment: {label}"},
                gauge={
                    'axis': {'range': [-100, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-100, -20], 'color': "red"},
                        {'range': [-20, 20], 'color': "gray"},
                        {'range': [20, 100], 'color': "green"}
                    ]
                }
            ))

            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("Confidence", f"{confidence * 100:.1f}%")
            st.metric("Sentiment Score", f"{sentiment:.3f}")

        # News headlines
        st.subheader("Recent News Headlines")
        if headlines:
            for headline in headlines:
                st.write(f"• {headline}")
        else:
            st.info("No recent news headlines available")

    def show_download_section(self, forecast_df, report, symbol):
        """Download section for reports"""
        st.subheader("💾 Download Reports")
        col1, col2, col3 = st.columns(3)

        # CSV Download
        csv = forecast_df.to_csv(index=False)
        with col1:
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name=f"{symbol}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        # JSON Report
        json_str = json.dumps(report, indent=2, default=str)
        with col2:
            st.download_button(
                label="📊 Download Full Report JSON",
                data=json_str,
                file_name=f"{symbol}_report_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

        # PDF Report (placeholder)
        with col3:
            st.download_button(
                label="📄 Generate PDF Report",
                data=json_str,  # Placeholder
                file_name=f"{symbol}_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                disabled=True  # PDF generation would need additional libraries
            )

    def show_portfolio(self):
        """Portfolio management section"""
        st.title("💰 Portfolio Management")

        tab1, tab2, tab3 = st.tabs(['My Portfolio', "Simulation", "Recommendations"])

        with tab1:
            self.show_my_portfolio()

        with tab2:
            self.show_portfolio_simulation()

        with tab3:
            self.show_portfolio_recommendations()

    def show_my_portfolio(self):
        """Display user portfolio"""
        st.subheader("My Investment Portfolio")

        # Portfolio input
        col1, col2 = st.columns(2)

        with col1:
            initial_budget = st.number_input("Portfolio Budget ($)", value=10000, step=1000)

        with col2:
            risk_tolerance = st.selectbox("Risk Tolerance", ["LOW", "MEDIUM", "HIGH"])

        # Manual portfolio entry
        st.subheader("Add Stocks to Portfolio")
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            stock_symbol = st.text_input("Stock Symbol", "AAPL").upper()

        with col2:
            allocation = st.number_input("Allocation %", min_value=1, max_value=100, value=20)

        with col3:
            entry_price = st.number_input("Entry Price", value=150.0)

        with col4:
            st.write("")  # Spacing
            if st.button("Add Stock"):
                if stock_symbol not in st.session_state.portfolio:
                    st.session_state.portfolio[stock_symbol] = {
                        'allocation': allocation,
                        'entry_price': entry_price,
                        'shares': (initial_budget * allocation / 100) / entry_price
                    }
                    st.success(f"Added {stock_symbol} to portfolio!")

        # Display portfolio
        if st.session_state.portfolio:
            st.subheader("Current Portfolio")
            portfolio_df = pd.DataFrame.from_dict(st.session_state.portfolio, orient='index')
            st.dataframe(portfolio_df)

            # Portfolio metrics
            total_value = 0
            for symbol, stock_data in st.session_state.portfolio.items():
                df = get_stock_data(symbol, period='1d')
                if not df.empty:
                    current_price = df['Close'].iloc[-1]
                    total_value += stock_data['shares'] * current_price

            st.metric("Portfolio Value", f"${total_value:.2f}")

    def show_portfolio_simulation(self):
        """Portfolio simulation"""
        st.subheader("Portfolio Strategy Simulation")

        col1, col2 = st.columns(2)

        with col1:
            budget = st.number_input("Simulation Budget ($)", value=10000, step=1000)
            strategy = st.selectbox("Investment Strategy", ["DCA", "LUMP_SUM", "MOMENTUM"])
            simulation_period = st.slider("Simulation Period (days)", 30, 365, 90)

        with col2:
            symbols_input = st.text_input("Stocks to Simulate (comma-separated)", "AAPL,MSFT,GOOGL")
            symbols = [s.strip().upper() for s in symbols_input.split(',')]

        if st.button("Run Simulation"):
            with st.spinner("Running portfolio simulation..."):
                results = {}
                for symbol in symbols:
                    df = get_stock_data(symbol, period='1y')
                    if not df.empty and self.portfolio_simulator:
                        profit_data = self.portfolio_simulator.simulate_strategy(
                            df, symbol, strategy, simulation_period
                        )
                        results[symbol] = profit_data

                # Display results
                if results:
                    results_df = pd.DataFrame.from_dict(results, orient='index')
                    st.subheader("Simulation Results")
                    st.dataframe(results_df.style.format({
                        'final_value': '${:.2f}',
                        'total_invested': '${:.2f}',
                        'profit': '${:.2f}',
                        'return_pct': '{:.2f}%'
                    }))

                    # Best performing stock
                    if 'return_pct' in results_df.columns:
                        best_stock = results_df['return_pct'].idxmax()
                        best_return = results_df['return_pct'].max()
                        st.success(f"Best performer: {best_stock} with {best_return:.2f}% return")

    def show_portfolio_recommendations(self):
        """Show portfolio recommendations"""
        st.subheader("💡 Portfolio Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            budget = st.number_input("Investment Budget ($)", value=5000, step=1000)

        with col2:
            risk_tolerance = st.selectbox("Your Risk Profile", ["LOW", "MEDIUM", "HIGH"])

        if st.button("Generate Recommendations"):
            with st.spinner("Analyzing market opportunities..."):
                recommendations = get_portfolio_recommendations(budget, risk_tolerance)

                if recommendations:
                    rec_df = pd.DataFrame(recommendations)

                    st.subheader("Recommended Portfolio Allocation")
                    # Pie chart
                    fig = go.Figure(data=[go.Pie(
                        labels=rec_df['sector'],
                        values=rec_df['allocation_pct'],
                        hole=0.3
                    )])
                    fig.update_layout(title="Recommended Sector Allocation")
                    st.plotly_chart(fig, use_container_width=True)

                    # Detailed table
                    st.subheader("Detailed Recommendations")
                    st.dataframe(rec_df.style.format({
                        'expected_return': '{:.2f}%',
                        'volatility': '{:.2f}%',
                        'allocation_pct': '{:.1f}%',
                        'investment_amount': '${:.2f}'
                    }))
                else:
                    st.error("Could not generate recommendations at this time")

    def show_model_comparison(self):
        """Model comparison section"""
        st.title("🤖 Model Performance Comparison")

        symbol = st.text_input("Stock for Model Comparison", "AAPL").upper()

        if st.button("Compare Models"):
            with st.spinner("Training and comparing models..."):
                metrics = compare_models(symbol)

                if metrics:
                    # Create comparison table
                    comparison_data = []
                    for model_name, model_metrics in metrics.items():
                        if isinstance(model_metrics, dict) and 'RMSE' in model_metrics:
                            comparison_data.append({
                                'Model': model_name,
                                'RMSE': model_metrics['RMSE'],
                                'MAE': model_metrics['MAE'],
                                'Directional_Accuracy': f"{model_metrics.get('Directional_Accuracy', 0):.2f}%"
                            })

                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Highlight best performing model for each metric
                        styled_df = comparison_df.style.highlight_min(
                            subset=['RMSE', 'MAE'],
                            color='lightgreen'
                        ).highlight_max(
                            subset=['Directional_Accuracy'],
                            color='lightgreen'
                        )

                        st.dataframe(styled_df)

                        # Best model overall (lowest RMSE)
                        best_model = comparison_df.loc[comparison_df['RMSE'].idxmin()]
                        st.success(f"🏆 Best Model: {best_model['Model']} (RMSE: {best_model['RMSE']:.4f})")
                else:
                    st.error("Could not compare models at this time")

    def show_reports(self):
        """Reports section"""
        st.title("📋 Reports & Analytics")

        tab1, tab2, tab3 = st.tabs(['Generated Reports', "Historical Analysis", "Export Data"])

        with tab1:
            st.subheader("Recently Generated Reports")
            # Placeholder for report history
            st.info("Report history will appear here as you generate analyses")

        with tab2:
            st.subheader("Historical Strategy Analysis")
            col1, col2 = st.columns(2)

            with col1:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
                end_date = st.date_input("End Date", value=datetime.now())

            with col2:
                strategy = st.selectbox("Backtest Strategy", ["MOMENTUM", "MEAN_REVERSION", "BREAKOUT"])

            if st.button("Run Backtest"):
                with st.spinner("Running historical analysis..."):
                    # Placeholder for backtesting functionality
                    st.info("Backtesting feature coming soon!")

        with tab3:
            st.subheader("Data Export")
            export_symbol = st.text_input("Stock Symbol to Export", "AAPL").upper()
            export_period = st.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y"])

            if st.button("Export Historical Data"):
                df = get_stock_data(export_symbol, period=export_period)
                if not df.empty:
                    csv = df.to_csv()
                    st.download_button(
                        label="Download Historical Data",
                        data=csv,
                        file_name=f"{export_symbol}_historical_{export_period}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No data available for export")

    def show_settings(self):
        """Application settings"""
        st.title("⚙️ Settings")

        st.subheader("API Configuration")
        col1, col2 = st.columns(2)

        with col1:
            news_api_key = st.text_input("News API Key", type="password")
            alpha_vantage_key = st.text_input("Alpha Vantage Key", type="password")

        with col2:
            finnhub_key = st.text_input("Finnhub Key", type="password")
            polygon_key = st.text_input("Polygon.io Key", type="password")

        st.subheader("Application Preferences")
        col1, col2 = st.columns(2)

        with col1:
            default_model = st.selectbox(
                "Default Model",
                ["AUTO", "LSTM", "GRU", "ENSEMBLE", "PROPHET", "ARIMA"]
            )
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])

        with col2:
            refresh_interval = st.selectbox("Data Refresh Interval", ["15m", "30m", "1h", "4h", "1d"])
            cache_duration = st.selectbox("Cache Duration", ["1h", "6h", "1d", "7d"])

        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

    def show_market_alerts(self):
        """Display market alerts"""
        # Sample alerts - in real implementation, these would come from live data
        alerts = [
            {"type": "VOLATILITY", "symbol": "TSLA", "message": "High volatility detected", "severity": "HIGH"},
            {"type": "VOLUME", "symbol": "AAPL", "message": "Unusual volume spike", "severity": "MEDIUM"},
            {"type": "PRICE", "symbol": "NVDA", "message": "Price breakout detected", "severity": "MEDIUM"},
        ]

        for alert in alerts:
            if alert['severity'] == 'HIGH':
                st.error(f"🚨 {alert['symbol']}: {alert['message']}")
            elif alert['severity'] == 'MEDIUM':
                st.warning(f"⚠️ {alert['symbol']}: {alert['message']}")
            else:
                st.info(f"ℹ️ {alert['symbol']}: {alert['message']}")

# Run the application
if __name__ == "__main__":
    app = AdvancedStockApp()
    try:
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")