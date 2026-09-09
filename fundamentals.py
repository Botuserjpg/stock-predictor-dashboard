# Replace the existing fundamentals.py with this enhanced version

import yfinance as yf
import pandas as pd
import requests
from typing import Dict, Any, List
import json
from datetime import datetime

class EnhancedFundamentalAnalyzer:
    """Enhanced fundamental analysis with multiple data sources"""
    
    def __init__(self):
        self.cache = {}
    
    def get_comprehensive_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive fundamental analysis"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Basic company info
            company_info = self._get_company_info(info, symbol)
            
            # Financial ratios
            valuation = self._get_valuation_metrics(info)
            profitability = self._get_profitability_metrics(info)
            liquidity = self._get_liquidity_metrics(info)
            efficiency = self._get_efficiency_metrics(info)
            growth = self._get_growth_metrics(info)
            
            # Financial statements
            financials = self._get_financial_statements(stock)
            
            # Analyst estimates
            analyst_data = self._get_analyst_estimates(stock)
            
            # Ownership data
            ownership = self._get_ownership_data(info)
            
            # ESG data if available
            esg_data = self._get_esg_data(info)
            
            # Composite scoring
            composite_score = self._calculate_composite_score(
                valuation, profitability, growth, analyst_data
            )
            
            return {
                'company_info': company_info,
                'valuation_metrics': valuation,
                'profitability_metrics': profitability,
                'liquidity_metrics': liquidity,
                'efficiency_metrics': efficiency,
                'growth_metrics': growth,
                'financial_statements': financials,
                'analyst_data': analyst_data,
                'ownership_data': ownership,
                'esg_data': esg_data,
                'composite_scores': composite_score,
                'analysis_date': datetime.now().isoformat(),
                'symbol': symbol
            }
            
        except Exception as e:
            return {'error': f'Failed to get fundamentals: {str(e)}'}
    
    def _get_company_info(self, info: Dict, symbol: str) -> Dict[str, Any]:
        """Get comprehensive company information"""
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'enterprise_value': info.get('enterpriseValue', 0),
            'employees': info.get('fullTimeEmployees', 0),
            'description': info.get('longBusinessSummary', ''),
            'country': info.get('country', 'Unknown'),
            'exchange': info.get('exchange', 'Unknown'),
            'currency': info.get('currency', 'USD'),
            'website': info.get('website', ''),
            'ceo': info.get('companyOfficers', [{}])[0].get('name', '') if info.get('companyOfficers') else ''
        }
    
    def _get_valuation_metrics(self, info: Dict) -> Dict[str, Any]:
        """Get comprehensive valuation metrics"""
        return {
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'price_to_book': info.get('priceToBook', 0),
            'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
            'enterprise_to_revenue': info.get('enterpriseToRevenue', 0),
            'enterprise_to_ebitda': info.get('enterpriseToEbitda', 0),
            'ev_to_ebit': info.get('enterpriseToEbit', 0),
            'market_cap_to_gdp': 0,  # Would need GDP data
            'shiller_pe': 0  # Would need historical data
        }
    
    def _get_profitability_metrics(self, info: Dict) -> Dict[str, Any]:
        """Get profitability metrics"""
        return {
            'gross_margin': info.get('grossMargins', 0),
            'operating_margin': info.get('operatingMargins', 0),
            'profit_margin': info.get('profitMargins', 0),
            'return_on_equity': info.get('returnOnEquity', 0),
            'return_on_assets': info.get('returnOnAssets', 0),
            'return_on_capital': info.get('returnOnCapital', 0),
            'ebitda_margin': info.get('ebitdaMargins', 0),
            'net_income_margin': info.get('netIncomeToCommon', 0) / info.get('totalRevenue', 1) if info.get('totalRevenue') else 0
        }
    
    def _get_liquidity_metrics(self, info: Dict) -> Dict[str, Any]:
        """Get liquidity and solvency metrics"""
        return {
            'current_ratio': info.get('currentRatio', 0),
            'quick_ratio': info.get('quickRatio', 0),
            'cash_ratio': info.get('totalCash', 0) / info.get('totalCurrentLiabilities', 1) if info.get('totalCurrentLiabilities') else 0,
            'debt_to_equity': info.get('debtToEquity', 0),
            'interest_coverage': info.get('ebitda', 0) / info.get('interestExpense', 1) if info.get('interestExpense') else 0,
            'debt_to_ebitda': info.get('totalDebt', 0) / info.get('ebitda', 1) if info.get('ebitda') else 0
        }
    
    def _get_efficiency_metrics(self, info: Dict) -> Dict[str, Any]:
        """Get operational efficiency metrics"""
        return {
            'asset_turnover': info.get('totalRevenue', 0) / info.get('totalAssets', 1) if info.get('totalAssets') else 0,
            'inventory_turnover': info.get('inventoryTurnover', 0),
            'receivables_turnover': info.get('receivablesTurnover', 0),
            'days_sales_outstanding': info.get('daysSalesOutstanding', 0),
            'days_inventory': info.get('daysOfInventoryOnHand', 0)
        }
    
    def _get_growth_metrics(self, info: Dict) -> Dict[str, Any]:
        """Get growth metrics"""
        return {
            'revenue_growth': info.get('revenueGrowth', 0),
            'earnings_growth': info.get('earningsGrowth', 0),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0),
            'ebitda_growth': 0,  # Would need historical data
            'free_cash_flow_growth': info.get('freeCashflow', 0) / abs(info.get('freeCashflow', 1)) if info.get('freeCashflow') else 0
        }
    
    def _get_financial_statements(self, stock) -> Dict[str, Any]:
        """Get financial statements"""
        try:
            # Income Statement
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            return {
                'income_statement': income_stmt.to_dict() if income_stmt is not None else {},
                'balance_sheet': balance_sheet.to_dict() if balance_sheet is not None else {},
                'cash_flow': cash_flow.to_dict() if cash_flow is not None else {}
            }
        except Exception:
            return {}
    
    def _get_analyst_estimates(self, stock) -> Dict[str, Any]:
        """Get analyst estimates and recommendations"""
        try:
            recommendations = stock.recommendations
            earnings_estimates = stock.earnings_estimates
            
            return {
                'recommendation_mean': stock.info.get('recommendationMean', 0),
                'recommendation_key': stock.info.get('recommendationKey', 'hold'),
                'number_of_analysts': stock.info.get('numberOfAnalystOpinions', 0),
                'target_mean_price': stock.info.get('targetMeanPrice', 0),
                'target_high_price': stock.info.get('targetHighPrice', 0),
                'target_low_price': stock.info.get('targetLowPrice', 0),
                'earnings_estimates': earnings_estimates.to_dict() if earnings_estimates is not None else {}
            }
        except Exception:
            return {}
    
    def _get_ownership_data(self, info: Dict) -> Dict[str, Any]:
        """Get ownership data"""
        return {
            'held_percent_insiders': info.get('heldPercentInsiders', 0),
            'held_percent_institutions': info.get('heldPercentInstitutions', 0),
            'short_percent_of_float': info.get('shortPercentOfFloat', 0),
            'short_ratio': info.get('shortRatio', 0)
        }
    
    def _get_esg_data(self, info: Dict) -> Dict[str, Any]:
        """Get ESG data if available"""
        return {
            'esg_score': info.get('esgScores', {}).get('totalEsg', 0),
            'environment_score': info.get('esgScores', {}).get('environmentScore', 0),
            'social_score': info.get('esgScores', {}).get('socialScore', 0),
            'governance_score': info.get('esgScores', {}).get('governanceScore', 0)
        }
    
    def _calculate_composite_score(self, valuation: Dict, profitability: Dict, 
                                 growth: Dict, analyst: Dict) -> Dict[str, Any]:
        """Calculate composite fundamental score"""
        scores = {}
        
        # Valuation score (lower is better)
        pe_score = min(100, max(0, (50 - valuation.get('pe_ratio', 50)) * 2))
        pb_score = min(100, max(0, (3 - valuation.get('price_to_book', 3)) * 33))
        scores['valuation_score'] = (pe_score + pb_score) / 2
        
        # Profitability score (higher is better)
        roe_score = min(100, profitability.get('return_on_equity', 0) * 10)
        margin_score = min(100, profitability.get('profit_margin', 0) * 100)
        scores['profitability_score'] = (roe_score + margin_score) / 2
        
        # Growth score (higher is better)
        revenue_growth_score = min(100, max(0, growth.get('revenue_growth', 0) * 500))
        earnings_growth_score = min(100, max(0, growth.get('earnings_growth', 0) * 100))
        scores['growth_score'] = (revenue_growth_score + earnings_growth_score) / 2
        
        # Analyst sentiment score
        analyst_score = 0
        if analyst.get('recommendation_mean'):
            # Convert recommendation to score (1=strong buy, 5=strong sell)
            rec_map = {1: 100, 2: 75, 3: 50, 4: 25, 5: 0}
            analyst_score = rec_map.get(round(analyst['recommendation_mean']), 50)
        scores['analyst_score'] = analyst_score
        
        # Overall composite score
        overall_score = (
            scores['valuation_score'] * 0.3 +
            scores['profitability_score'] * 0.25 +
            scores['growth_score'] * 0.25 +
            scores['analyst_score'] * 0.2
        )
        scores['composite_score'] = overall_score
        
        # Rating based on composite score
        if overall_score >= 80:
            scores['rating'] = 'STRONG_BUY'
        elif overall_score >= 60:
            scores['rating'] = 'BUY'
        elif overall_score >= 40:
            scores['rating'] = 'HOLD'
        elif overall_score >= 20:
            scores['rating'] = 'SELL'
        else:
            scores['rating'] = 'STRONG_SELL'
        
        return scores

# Update the main function
def get_company_fundamentals(symbol: str) -> Dict[str, Any]:
    """Get comprehensive company fundamentals"""
    analyzer = EnhancedFundamentalAnalyzer()
    return analyzer.get_comprehensive_fundamentals(symbol)