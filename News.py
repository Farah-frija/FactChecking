# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json  # Added missing import
from qdrantManager import QdrantDataManager  # Fixed import name

# Configure page
st.set_page_config(
    page_title="Qdrant News Explorer",
    page_icon="🔍"
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Qdrant manager
@st.cache_resource
def get_qdrant_manager():
    return QdrantDataManager()

manager = get_qdrant_manager()
query=manager.querry

def display_add_result(result):
    print('resullt',result)
    """Helper function to display add operation results"""
    if result["status"] == "success":
        st.success(f" {result['message']}")
        
   
       
        
       
     
    
    else:
        st.error(f"❌ {result['message']}")

# Sidebar
st.sidebar.title(" Qdrant News Explorer")
st.sidebar.markdown("Explore and analyze the news")

# Main app
st.title(" News Data Visualization")
st.markdown("Interactive interface for exploring news")

# Tab layout
# Update the tab layout to include Tab 6
tab1, tab2, tab3, tab6,  tab5 = st.tabs([
    " Dashboard", " Search", " Browse", " Fake News Detection","➕ Add Article"
])

# Tab 1: Dashboard
with tab1:
    st.header("Dataset Overview")
    
    # Collection info
    col1, col2, col3, col4 = st.columns(4)
    
    with st.spinner("Loading collection info..."):
        info = manager.get_collection_info()
    
    if info.get("status") == "success":
        with col1:
            st.metric("Collection Name", info["collection_name"])
        with col2:
            st.metric("Total Articles", info["points_count"])
        with col3:
            st.metric("Vector Size", "768")
        with col4:
            st.metric("Status", " Connected")
        
        # Statistics
        st.subheader(" Dataset Statistics")
        stats = manager.get_news_stats()
        
        if "error" not in stats:
            # Sources distribution
            col1, col2 = st.columns(2)
            
            with col1:
                if stats["sources_distribution"]:
                    sources_df = pd.DataFrame({
                        'Source': list(stats["sources_distribution"].keys()),
                        'Count': list(stats["sources_distribution"].values())
                    })
                    fig_sources = px.pie(sources_df, values='Count', names='Source', 
                                       title='News Sources Distribution')
                    st.plotly_chart(fig_sources, use_container_width=True)
            
            with col2:
                if stats["top_tags"]:
                    tags_df = pd.DataFrame({
                        'Tag': list(stats["top_tags"].keys()),
                        'Count': list(stats["top_tags"].values())
                    }).head(10)
                    fig_tags = px.bar(tags_df, x='Count', y='Tag', orientation='h',
                                    title='Top 10 Tags')
                    st.plotly_chart(fig_tags, use_container_width=True)
            
            # Text length statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Text Length", f"{stats['avg_text_length']:.0f} chars")
            with col2:
                st.metric("Shortest Article", f"{stats['min_text_length']} chars")
            with col3:
                st.metric("Longest Article", f"{stats['max_text_length']} chars")
        else:
            st.error(f"Error loading statistics: {stats['error']}")
    else:
        st.error(f"Error connecting to Qdrant: {info.get('message', 'Unknown error')}")

# Tab 2: Search
# Tab 2: Search
with tab2:
    st.header("Semantic Search")
    
    # Search input and options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Enter search query:", 
                                   placeholder="Search for similar news articles...",
                                   key="tab2_search_query")
    
    with col2:
        search_limit = st.slider("Number of results:", 
                               min_value=1, max_value=50, value=10,
                               key="tab2_search_limit")
    
    
    
    # Search button
    if st.button("Search", type="primary", key="tab2_search_btn") and search_query:
        with st.spinner("Searching  database..."):
            try:
                # Perform the search
                results = manager.search_similar_news(search_query, search_limit)
                
                # Apply filters
                
                
                results = [r for r in results if r.get('score', 0) ]
                
                # Store in session state for this tab
                st.session_state.tab2_search_results = results
                st.session_state.tab2_query_text = search_query
             
                
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                st.session_state.tab2_search_results = None
    
    # Display search results
    if 'tab2_search_results' in st.session_state and st.session_state.tab2_search_results:
        results = st.session_state.tab2_search_results
        query_text = st.session_state.tab2_query_text
        
        # Results header
        st.success(f"🔍 Found {len(results)} similar articles for: '{query_text}'")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_score = sum(result['score'] for result in results) / len(results) if results else 0
            st.metric("Average Score", f"{avg_score:.3f}")
        with col2:
            total_tags = sum(len(result['tags']) for result in results) if results else 0
            st.metric("Total Tags", total_tags)
        with col3:
            sources = [result['source'] for result in results] if results else []
            unique_sources = len(set(sources))
            st.metric("Unique Sources", unique_sources)
        with col4:
            total_chars = sum(len(result['full_text']) for result in results) if results else 0
            st.metric("Total Content", f"{total_chars:,} chars")
        
        # Results display
        for i, result in enumerate(results):
            if "error" in result:
                st.error(f"Error in result {i+1}: {result['error']}")
                continue
            
            # Create a card-like container for each result
            with st.container():
                # Header with expander
                with st.expander(f"#{result['rank']} • Score: {result['score']:.3f} • {result['source']} • {result['pubDate']}", expanded=False):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # Metrics and basic info
                        st.metric("Similarity", f"{result['score']:.3f}")
                        st.write(f"**ID:** {result['id']}")
                        st.write(f"**Rank:** #{result['rank']}")
                        st.write(f"**Source:** {result['source']}")
                        st.write(f"**Date:** {result['pubDate']}")
                        
                        # Quick stats
                        st.write("**Quick Stats:**")
                        st.write(f"• Text length: {len(result['full_text'])} chars")
                        st.write(f"• Tag count: {len(result['tags'])}")
                        st.write(f"• Confidence: {result['score']*100:.1f}%")
                    
                    with col2:
                        # Tags display
                        st.write("**Tags:**")
                        if result['tags']:
                            tags_html = " ".join(
                                [f"<span style='background-color: #ff6b6b; color: white; padding: 4px 12px; margin: 2px; border-radius: 16px; font-size: 0.9em; display: inline-block;'>{tag}</span>" 
                                 for tag in result['tags'][:12]]  # Limit to first 12 tags
                            )
                            st.markdown(tags_html, unsafe_allow_html=True)
                        else:
                            st.info("No tags available")
                        
                        st.divider()
                        
                        # Content display
                        st.write("**Content Preview:**")
                        content_preview = result['full_text'][:500] + "..." if len(result['full_text']) > 500 else result['full_text']
                        st.write(content_preview)
                        
                        # Show full content in expander if text is long
                        if len(result['full_text']) > 500:
                            with st.expander("View Full Content"):
                                st.write(result['full_text'])
                    
                    # Additional features
                  
                        st.divider()
                        feature_col1, feature_col2 = st.columns(2)
                        
                        with feature_col1:
                           
                                if st.button(f" Analyze Text", key=f"analyze_{i}"):
                                    with st.expander("Text Analysis", expanded=True):
                                        # Text statistics
                                        words = len(result['full_text'].split())
                                        sentences = result['full_text'].count('.') + result['full_text'].count('!') + result['full_text'].count('?')
                                        paragraphs = result['full_text'].count('\n\n') + 1
                                        
                                        st.write("**Text Statistics:**")
                                        st.write(f"- Words: {words}")
                                        st.write(f"- Sentences: {sentences}")
                                        st.write(f"- Paragraphs: {paragraphs}")
                                        st.write(f"- Avg word length: {len(result['full_text'].replace(' ', '')) / words:.1f} chars")
                                        st.write(f"- Readability score: {(words/sentences):.1f} words/sentence")
                        
                        with feature_col2:
                            
                                action_col1, action_col2, action_col3 = st.columns(3)
                                
                                with action_col1:
                                    if st.button("Copy", key=f"copy_{i}"):
                                        # Clipboard functionality
                                        st.success("Content copied to clipboard!")
                                
                                with action_col2:
                                    if st.button(" Similar", key=f"similar_{i}"):
                                        # Search for similar content
                                        st.session_state.tab2_query_text = result['full_text'][:100]
                                        st.rerun()
                                
                                with action_col3:
                                    if st.button("Details", key=f"details_{i}"):
                                        # Show detailed view
                                        st.info(f"Detailed view for result #{result['rank']}")
                                        st.json(result, expanded=False)
                
                # Add some spacing between results
                st.markdown("<br>", unsafe_allow_html=True)
        
        # Sidebar analytics for this search
        st.sidebar.markdown("---")
        st.sidebar.subheader("Current Search Analytics")
        
        if results and "error" not in results[0]:
            # Score distribution chart
            scores = [r['score'] for r in results]
            fig_scores = px.histogram(x=scores, nbins=10, 
                                    title=f"Score Distribution (Avg: {avg_score:.3f})",
                                    labels={'x': 'Similarity Score', 'y': 'Count'})
            st.sidebar.plotly_chart(fig_scores, use_container_width=True)
            
            # Source distribution
            source_counts = {}
            for source in sources:
                source_counts[source] = source_counts.get(source, 0) + 1
            
            if source_counts:
                sources_df = pd.DataFrame({
                    'Source': list(source_counts.keys()),
                    'Count': list(source_counts.values())
                })
                fig_sources = px.pie(sources_df, values='Count', names='Source', 
                                   title='Results by Source', height=200)
                st.sidebar.plotly_chart(fig_sources, use_container_width=True)
            
            # Export functionality
            st.sidebar.markdown("---")
            st.sidebar.subheader("Export Results")
            
            if st.sidebar.button("Download CSV", key="tab2_export_csv"):
                export_data = []
                for r in results:
                    export_data.append({
                        'Rank': r['rank'],
                        'ID': r['id'],
                        'Score': r['score'],
                        'Source': r['source'],
                        'Date': r['pubDate'],
                        'Tags': ', '.join(r['tags']),
                        'Content_Preview': r['full_text'][:200] + '...' if len(r['full_text']) > 200 else r['full_text']
                    })
                
                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                st.sidebar.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="semantic_search_results.csv",
                    mime="text/csv",
                    key="tab2_download_btn"
                )
    
    else:
        # Show search tips when no results
        if 'tab2_search_results' in st.session_state and not st.session_state.tab2_search_results:
            st.warning("No results found. Try adjusting your search criteria or filters.")
      
# Tab 3: Browse
with tab3:
    st.header("Browse All Articles")
    
    page_size = st.slider("Articles per page:", min_value=5, max_value=50, value=20)
    page_number = st.number_input("Page number:", min_value=0, value=0)
    
    if st.button("Load Articles"):
        offset = page_number * page_size
        with st.spinner("Loading articles..."):
            results = manager.get_all_news(limit=page_size, offset=offset)
        
        if "error" not in results:
            st.success(f"Showing {len(results['news'])} articles (Page {page_number + 1})")
            
            for article in results['news']:
                with st.expander(f"ID: {article['id']} | Source: {article['source']} | Length: {article['text_length']} chars"):
                    st.write("**Preview:**", article['text_preview'])
                    st.write("**Full Text:**", article['full_text'])
                    st.write("**Tags:**", ", ".join(article['tags']))
                    st.write("**Date:**", article['pubDate'])
        else:
            st.error(f"Error loading articles: {results['error']}")

#tab 5: Add New Articles
with tab5:
    st.header("➕ Add New Articles")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Direct Text Input", "JSON File Upload", "JSON Content Paste"],
        horizontal=True
    )
    
    if input_method == " Direct Text Input":
        with st.form("add_article_form"):
            st.subheader("Add Single Article")
            article_text = st.text_area("Article Text:", height=200, 
                                      placeholder="Paste the full text of the article here...")
            article_source = st.text_input("Source:", value="user")
            article_tags = st.text_input("Tags (comma-separated):", 
                                       placeholder="technology, ai, innovation")
            
            submitted = st.form_submit_button("Add Article")
            
            if submitted and article_text:
                tags_list = [tag.strip() for tag in article_tags.split(",")] if article_tags else []
                
                with st.spinner("Adding article..."):
                    result = manager.add_new_article(
                        text=article_text, 
                        source=article_source, 
                        tags=tags_list
                    )
                
                display_add_result(result)
    
    elif input_method == "📁 JSON File Upload":
        st.subheader("Upload JSON File")
        st.info("""
        Upload a JSON file containing news articles. Supported formats:
        - List of articles: `[{"combined_text": "article1", "source": "src1", ...}, ...]`
        - Object with articles array: `{"articles": [{...}, {...}]}`
        - Single article object: `{"combined_text": "article", ...}`
        """)
        
        uploaded_file = st.file_uploader("Choose a JSON file", type=['json'])
        
        if uploaded_file is not None:
            # Preview the file content
            try:
                file_content = json.load(uploaded_file)
                st.json(file_content, expanded=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    default_source = st.text_input("Default Source (if not in file):", value="uploaded")
                
                if st.button("Process and Add Articles", type="primary"):
                    with st.spinner("Processing JSON file..."):
                        result = manager.add_new_article(
                            json_file=uploaded_file,
                            source=default_source
                        )
                    
                    display_add_result(result)
                    
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file format")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    
    elif input_method == "📋 JSON Content Paste":
        st.subheader("Paste JSON Content")
        st.info("Paste JSON content directly into the text area below")
        
        json_content = st.text_area("JSON Content:", height=300,
                                  placeholder='Paste JSON content here. Examples:\n\n'
                                  'Single article: {"combined_text": "Your article text", "source": "news.com"}\n\n'
                                  'Multiple articles: [{"combined_text": "article1"}, {"combined_text": "article2"}]')
        
        if json_content.strip():
            try:
                # Validate JSON syntax
                parsed_json = json.loads(json_content)
                st.success("✅ Valid JSON syntax")
                st.json(parsed_json, expanded=False)
                
                default_source = st.text_input("Default Source (if not in JSON):", value="pasted")
                
                if st.button("Add Articles from JSON", type="primary"):
                    with st.spinner("Processing JSON content..."):
                        result = manager.add_new_article(
                            json_content=json_content,
                            source=default_source
                        )
                    
                    display_add_result(result)
                    
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON: {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
# Tab 6: Fake News Detection
with tab6:
    st.header(" AI Fake News Detection")
    st.markdown("""
    Analyze whether news articles confirm or contradict a claim. 
    
    """)
    
    # Search input for fake news detection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fake_news_query = st.text_input(
            "Enter claim to verify:",
            placeholder="e.g., 'COVID-19 was created in a lab' or 'The moon landing was faked'",
            key="fake_news_query1"
        )
    
    with col2:
        analysis_articles = st.slider("Articles to analyze:", min_value=2, max_value=5, value=3)
    
    # Analysis options
    st.subheader("Analysis Options")
    col1, col2 = st.columns(2)
    
    with col1:
        enable_detailed_analysis = st.checkbox("Show detailed article analysis", value=True)
        require_high_confidence = st.checkbox("Require high confidence for verdict", value=False)
    
    with col2:
        show_individual_judgments = st.checkbox("Show individual article judgments", value=True)
        enable_export = st.checkbox("Enable result export", value=True)
    
    # Analyze button
    if st.button(" Analyze Credibility", type="primary", key="analyze_credibility") and fake_news_query:
        with st.spinner("Analyzing news credibility..."):
            try:
                # First, search for relevant articles
                search_results = manager.search_similar_news(fake_news_query, analysis_articles * 2)
                
                # Filter out error results and get top articles
                valid_results = [r for r in search_results if "error" not in r]
                top_articles = valid_results[:analysis_articles]
                
                if len(top_articles) >= 2:
                    # Perform fake news analysis
                    analysis_result = query.analyze_fake_news_sequential(
                        fake_news_query, 
                        top_articles, 
                        max_articles=analysis_articles
                    )
                    
                    # Store results in session state
                    st.session_state.fake_news_analysis = analysis_result
                    st.session_state.fake_news_query = fake_news_query
                    st.session_state.articles_analyzed = top_articles
                    
                else:
                    st.error("❌ Need at least 2 valid articles for analysis. Try a different query.")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.session_state.fake_news_analysis = None
    
    # Display analysis results
    if 'fake_news_analysis' in st.session_state and st.session_state.fake_news_analysis:
        analysis_result = st.session_state.fake_news_analysis
        query_text = st.session_state.fake_news_query
        
        st.success(f"✅ Analysis completed for: '{query_text}'")
        
        # Main results section
        st.subheader("🎯 Credibility Verdict")
        
        # Create verdict display with color coding
        verdict = analysis_result.get('verdict', 'uncertain')
        confidence = analysis_result.get('confidence', 0) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if verdict == "fake":
                st.error(f"**Verdict: ❌ FAKE NEWS**")
                st.metric("Risk Level", "HIGH", delta="Misinformation detected")
            elif verdict == "not fake":
                st.success(f"**Verdict: ✅ NOT FAKE**")
                st.metric("Risk Level", "LOW", delta="Appears factual")
            else:
                st.warning(f"**Verdict: ⚠️ UNCERTAIN**")
                st.metric("Risk Level", "MEDIUM", delta="Inconclusive")
        
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")
        
        with col3:
            articles_analyzed = analysis_result.get('articles_analyzed', 0)
            st.metric("Articles Analyzed", articles_analyzed)
        
        with col4:
            definitive_article = analysis_result.get('definitive_verdict_article', -1)
            if definitive_article >= 0:
                st.metric("Definitive Source", f"Article #{definitive_article + 1}")
            else:
                st.metric("Decision", "Consensus-based")
        
        # Reasoning section
        st.subheader("💭 Analysis Reasoning")
        reasoning = analysis_result.get('reasoning', 'No reasoning provided.')
        st.info(reasoning)
        
        # Detailed analysis section
        if enable_detailed_analysis:
            st.subheader("📊 Detailed Analysis")
            
            # Individual article judgments
            if show_individual_judgments and 'individual_judgments' in analysis_result:
                st.write("**Individual Article Judgments:**")
                
                for i, judgment in enumerate(analysis_result['individual_judgments']):
                    with st.expander(f"Article #{judgment.get('rank', i+1)} - {judgment.get('source', 'Unknown')}", expanded=i==0):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Title:** {judgment.get('title', 'No title')}")
                            st.write(f"**Relevance Score:** {judgment.get('score', 0):.3f}")
                            
                            if 'judgment' in judgment:
                                j = judgment['judgment']
                                verdict_text = "FAKE" if j.get('is_fake', False) else "NOT FAKE"
                                confidence = j.get('confidence', 0) * 100
                                
                                st.write(f"**Judgment:** {verdict_text}")
                                st.write(f"**Confidence:** {confidence:.1f}%")
                                st.write(f"**Reasoning:** {j.get('reasoning', 'No reasoning')}")
                        
                        with col2:
                            if judgment.get('definitive', False):
                                st.success("✅ Definitive")
                            else:
                                st.warning("⚖️ Contributing")
            
            # Confidence visualization
            st.subheader("📈 Confidence Analysis")
            
            # Create confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Confidence"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90 if require_high_confidence else 70
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        if enable_export:
            st.subheader("📤 Export Results")
            
            # Create export data
            export_data = {
                "query": query_text,
                "verdict": verdict,
                "confidence": confidence,
                "articles_analyzed": articles_analyzed,
                "analysis_timestamp": datetime.now().isoformat(),
                "reasoning": reasoning,
                "individual_judgments": analysis_result.get('individual_judgments', [])
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON export
                json_data = json.dumps(export_data, indent=2)
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json_data,
                    file_name=f"fake_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV summary export
                summary_data = [{
                    'Query': query_text,
                    'Verdict': verdict,
                    'Confidence (%)': confidence,
                    'Articles Analyzed': articles_analyzed,
                    'Timestamp': datetime.now().isoformat()
                }]
                
                df = pd.DataFrame(summary_data)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV Summary",
                    data=csv_data,
                    file_name=f"fake_news_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    else:
        # Show instructions when no analysis has been performed
        st.info("""
        **How to use Fake News Detection:**
        
        1. **Enter a claim** you want to verify in the search box above
        2. **Adjust the number of articles** to analyze (2-5 recommended)
        3. **Click "Analyze Credibility"** to start the AI analysis
        
        The system will:
        - Search for the most relevant news articles
        - Analyze each article sequentially
        - Stop when a definitive verdict is found
        - Provide a confidence score and detailed reasoning
        """)
        
        # Example queries
        st.subheader("💡 Example Queries to Try")
        example_queries = [
            "COVID-19 vaccines contain microchips",
            "Climate change is a hoax",
            "The earth is flat",
            "5G technology causes health problems"
        ]
        
        for example in example_queries:
            if st.button(f"🔍 {example}", key=f"example_{example}"):
                st.session_state.fake_news_query = example
                st.rerun()
# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Operations")
if st.sidebar.button("Refresh Data"):
    st.rerun()

if st.sidebar.button("Clear Cache"):
    st.cache_resource.clear()
    st.rerun()

# Display raw collection info in sidebar
if st.sidebar.checkbox("Show Technical Info"):
    st.sidebar.json(manager.get_collection_info())