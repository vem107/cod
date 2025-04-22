from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import networkx as nx

# Initialize Spark Session
def init_spark():
    """Initialize Spark session"""
    spark = SparkSession.builder \
        .appName("Transaction Data Analysis") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    print("Spark session initialized")
    return spark

# Load the data
def load_data(spark, file_path):
    """Load transaction data from CSV file using Spark"""
    try:
        # Define schema to ensure proper data types
        schema = StructType([
            StructField("Invoice", StringType(), True),
            StructField("StockCode", StringType(), True),
            StructField("Description", StringType(), True),
            StructField("Quantity", IntegerType(), True),
            StructField("InvoiceDate", TimestampType(), True),
            StructField("Price", DoubleType(), True),
            StructField("Customer ID", DoubleType(), True),
            StructField("Country", StringType(), True)
        ])
        
        # Read CSV file
        df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load(file_path)
        
        # Convert to timestamp and calculate total amount
        df = df.withColumn("InvoiceDate", F.to_timestamp(F.col("InvoiceDate"))) \
            .withColumn("TotalAmount", F.col("Quantity") * F.col("Price"))
        
        print(f"Loaded {df.count()} transactions successfully")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Basic data statistics
def generate_basic_stats(df):
    """Generate basic statistics about the transaction data using Spark"""
    # Count unique values
    total_transactions = df.select("Invoice").distinct().count()
    total_products = df.select("StockCode").distinct().count()
    total_customers = df.select("Customer ID").distinct().count()
    total_countries = df.select("Country").distinct().count()
    
    # Get date range
    date_range = df.agg(
        F.min("InvoiceDate").alias("min_date"),
        F.max("InvoiceDate").alias("max_date")
    ).collect()[0]
    
    # Calculate total revenue
    total_revenue = df.agg(F.sum("TotalAmount").alias("total_revenue")).collect()[0]["total_revenue"]
    
    stats = {
        'Total Transactions': total_transactions,
        'Total Products': total_products,
        'Total Customers': total_customers,
        'Total Countries': total_countries,
        'Date Range': f"{date_range['min_date'].date()} to {date_range['max_date'].date()}",
        'Total Revenue': f"${total_revenue:.2f}"
    }
    return stats

# Graph analytics - build customer-product network
def build_customer_product_graph(df):
    """Build a bipartite graph of customers and products from Spark DataFrame"""
    # Convert to pandas for network analysis
    # Group by Customer ID and StockCode to get edges
    grouped_df = df.filter(F.col("Customer ID").isNotNull()) \
        .groupBy("Customer ID", "StockCode") \
        .agg(F.sum("Quantity").alias("Quantity")) \
        .toPandas()
    
    # Get unique customers and products
    customers = df.filter(F.col("Customer ID").isNotNull()) \
        .select("Customer ID").distinct().toPandas()
    products = df.select("StockCode").distinct().toPandas()
    
    # Build graph
    G = nx.Graph()
    
    # Add customer nodes
    for customer in customers["Customer ID"]:
        G.add_node(f"C-{customer}", type="customer")
    
    # Add product nodes
    for product in products["StockCode"]:
        G.add_node(f"P-{product}", type="product")
    
    # Add edges
    for _, row in grouped_df.iterrows():
        G.add_edge(f"C-{row['Customer ID']}", f"P-{row['StockCode']}", weight=row['Quantity'])
    
    return G

# Visualize customer-product graph (same as original)
def visualize_graph(G, max_nodes=100):
    """Visualize the graph with the most connected nodes"""
    if len(G) > max_nodes:
        # Get top nodes by degree
        nodes_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        top_nodes = [n[0] for n in nodes_by_degree]
        H = G.subgraph(top_nodes)
    else:
        H = G
    
    # Define node colors by type
    colors = ['skyblue' if 'C-' in node else 'lightgreen' for node in H.nodes()]
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(H, seed=42)
    nx.draw(H, pos, node_color=colors, with_labels=False, 
            node_size=50, edge_color='gray', alpha=0.7)
    plt.title("Customer-Product Network (Most Connected Nodes)")
    return plt.gcf()

# Generate reports
def generate_reports(df):
    """Generate various reports from the transaction data using Spark"""
    reports = {}
    
    # 1. Top selling products
    top_products = df.groupBy("StockCode") \
        .agg(
            F.first("Description").alias("Description"),
            F.sum("Quantity").alias("Quantity"),
            F.sum("TotalAmount").alias("TotalAmount")
        ) \
        .orderBy(F.desc("TotalAmount")) \
        .limit(10) \
        .toPandas()
    
    reports['top_products'] = top_products.set_index("StockCode")
    
    # 2. Sales by country
    sales_by_country = df.groupBy("Country") \
        .agg(
            F.sum("TotalAmount").alias("TotalAmount"),
            F.countDistinct("Invoice").alias("Orders"),
            F.countDistinct("Customer ID").alias("Unique Customers")
        ) \
        .orderBy(F.desc("TotalAmount")) \
        .toPandas() \
        .set_index("Country")
    
    reports['sales_by_country'] = sales_by_country
    
    # 3. Monthly sales trends
    monthly_sales = df.withColumn("Month", F.date_format("InvoiceDate", "yyyy-MM")) \
        .groupBy("Month") \
        .agg(
            F.sum("TotalAmount").alias("TotalAmount"),
            F.countDistinct("Invoice").alias("Orders"),
            F.countDistinct("Customer ID").alias("Unique Customers")
        ) \
        .orderBy("Month") \
        .toPandas() \
        .set_index("Month")
    
    reports['monthly_sales'] = monthly_sales
    
    # 4. Customer segmentation by purchase volume
    # First get customer totals
    customer_segments = df.groupBy("Customer ID") \
        .agg(
            F.sum("TotalAmount").alias("TotalAmount"),
            F.countDistinct("Invoice").alias("Orders")
        ) \
        .withColumn("AvgOrderValue", F.col("TotalAmount") / F.col("Orders")) \
        .toPandas()
    
    # Define segments
    def categorize(row):
        if row['TotalAmount'] > 5000:
            return 'High Value'
        elif row['TotalAmount'] > 1000:
            return 'Medium Value'
        else:
            return 'Low Value'
    
    customer_segments['Segment'] = customer_segments.apply(categorize, axis=1)
    
    # Summarize segments
    segment_summary = customer_segments.groupby('Segment').agg({
        'Customer ID': 'count',
        'TotalAmount': 'sum',
        'AvgOrderValue': 'mean'
    }).rename(columns={'Customer ID': 'Count'})
    
    reports['customer_segments'] = segment_summary
    
    return reports

# Visualize reports (same as original)
def visualize_reports(df, reports):
    """Create visualizations for the reports"""
    visualizations = {}
    
    # 1. Top products bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    top10 = reports['top_products'].sort_values('TotalAmount')
    sns.barplot(x='TotalAmount', y=top10.index, data=top10, ax=ax)
    ax.set_title('Top 10 Products by Revenue')
    ax.set_xlabel('Total Revenue')
    ax.set_ylabel('Stock Code')
    visualizations['top_products'] = fig
    
    # 2. Sales by country bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    top_countries = reports['sales_by_country'].head(10).sort_values('TotalAmount')
    sns.barplot(x='TotalAmount', y=top_countries.index, data=top_countries, ax=ax)
    ax.set_title('Revenue by Top 10 Countries')
    ax.set_xlabel('Total Revenue')
    ax.set_ylabel('Country')
    visualizations['sales_by_country'] = fig
    
    # 3. Monthly sales line chart
    fig, ax = plt.subplots(figsize=(12, 6))
    reports['monthly_sales']['TotalAmount'].plot(ax=ax, marker='o')
    ax.set_title('Monthly Sales Trend')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Revenue')
    ax.grid(True)
    visualizations['monthly_sales'] = fig
    
    # 4. Customer segment analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    reports['customer_segments']['TotalAmount'].plot.bar(ax=ax, color=['green', 'orange', 'blue'])
    ax.set_title('Revenue by Customer Segment')
    ax.set_ylabel('Total Revenue')
    ax.set_xlabel('Customer Segment')
    visualizations['customer_segments'] = fig
    
    # 5. Heatmap of product co-occurrence
    try:
        # Convert to pandas for heatmap creation
        pandas_df = df.toPandas()
        
        # Get top 20 products
        top_products = pandas_df.groupby('StockCode')['Quantity'].sum().nlargest(20).index
        
        # Filter data for these products
        filtered_df = pandas_df[pandas_df['StockCode'].isin(top_products)]
        
        # Create a pivot table for co-occurrence
        invoice_items = filtered_df.groupby(['Invoice', 'StockCode'])['Quantity'].sum().reset_index()
        invoice_items['Value'] = 1
        product_matrix = invoice_items.pivot_table(index='Invoice', columns='StockCode', values='Value', fill_value=0)
        
        # Calculate co-occurrence matrix
        co_occurrence = product_matrix.T.dot(product_matrix)
        np.fill_diagonal(co_occurrence.values, 0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(co_occurrence, cmap='YlGnBu', ax=ax)
        ax.set_title('Product Co-occurrence Heatmap (Top 20 Products)')
        visualizations['product_co_occurrence'] = fig
    except Exception as e:
        print(f"Error creating co-occurrence heatmap: {e}")
    
    return visualizations

# Generate PDF report (same as original)
def export_to_pdf(stats, visualizations, reports, output_file="transaction_analysis_report.pdf"):
    """Export all visualizations and reports to a PDF file"""
    with PdfPages(output_file) as pdf:
        # Cover page
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        ax.text(0.5, 0.8, 'Transaction Data Analysis Report', fontsize=24, ha='center')
        ax.text(0.5, 0.7, f"Generated on {datetime.now().strftime('%Y-%m-%d')}", fontsize=14, ha='center')
        ax.text(0.5, 0.5, "Summary Statistics:", fontsize=18, ha='center')
        
        y_pos = 0.4
        for key, value in stats.items():
            ax.text(0.5, y_pos, f"{key}: {value}", fontsize=12, ha='center')
            y_pos -= 0.05
            
        pdf.savefig(fig)
        plt.close(fig)
        
        # Add all visualizations
        for title, fig in visualizations.items():
            pdf.savefig(fig)
            plt.close(fig)
        
        # Add network graph if it exists
        if 'network_graph' in visualizations:
            pdf.savefig(visualizations['network_graph'])
        
        # Add tabular reports
        for report_name, report_df in reports.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axis('off')
            ax.text(0.5, 0.95, f"Report: {report_name.replace('_', ' ').title()}", fontsize=16, ha='center')
            
            # Convert DataFrame to a table
            table_data = report_df.reset_index()
            # Limit number of rows for display if too many
            if len(table_data) > 20:
                table_data = table_data.head(20)
                
            table = ax.table(
                cellText=table_data.values,
                colLabels=table_data.columns,
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            pdf.savefig(fig)
            plt.close(fig)
            
    print(f"Report exported to {output_file}")
    return output_file

# Main function
def analyze_transaction_data(file_path):
    """Main function to analyze transaction data and generate reports using Spark"""
    print(f"Attempting to load data from: {file_path}")
    
    # Initialize Spark
    spark = init_spark()
    
    # Load data
    df = load_data(spark, file_path)
    if df is None:
        print(f"Failed to load data from {file_path}. Please check the file path.")
        return
    
    # Generate basic statistics
    stats = generate_basic_stats(df)
    print("Basic Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cache the dataframe for better performance
    df.cache()
    
    # Build graph for network analysis
    G = build_customer_product_graph(df)
    print(f"Built graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Generate network graph visualization
    network_graph = visualize_graph(G)
    
    # Generate reports
    reports = generate_reports(df)
    print("Generated reports:")
    for report_name in reports.keys():
        print(f"  - {report_name}")
    
    # Create visualizations
    visualizations = visualize_reports(df, reports)
    visualizations['network_graph'] = network_graph
    print("Created visualizations")
    
    # Export to PDF
    output_file = export_to_pdf(stats, visualizations, reports)
    print(f"Analysis complete. Report saved to {output_file}")
    
    # Stop Spark session
    spark.stop()
    
    return {
        'stats': stats,
        'reports': reports,
        'visualizations': visualizations,
        'graph': G,
        'output_file': output_file
    }

if __name__ == "__main__":
    # Replace with your actual file path
    analyze_transaction_data("")
