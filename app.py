# Import necessary modules
import asyncio
import os
import torch
import streamlit as st

torch.classes.__path__ = []  # Neutralizes the path inspection
from querry_engine import QueryEngine

import streamlit.components.v1 as components  # For embedding custom HTML
import generate_knowledge_graph as KG

# Set up Streamlit page configuration
st.set_page_config(
    page_icon=None, 
    layout="wide",  # Use wide layout for better graph display
    initial_sidebar_state="auto", 
    menu_items=None
)

# Set the title of the app
st.title("Knowledge Graph From Text")

# Initialize session state variables
if 'graph_generated' not in st.session_state:
    st.session_state.graph_generated = False
if 'graph_html' not in st.session_state:
    st.session_state.graph_html = None
if 'enhanced_docs' not in st.session_state:
    st.session_state.enhanced_docs = None

# Sidebar section for user input method
start = st.text_input("Enter the starting line:", placeholder="Type here...")
end = st.text_input("Enter the ending line:", placeholder="Type here...")

# Generate Knowledge Graph button
if st.sidebar.button("Generate Knowledge Graph"):
    with st.spinner("Generating knowledge graph..."):
        # Call the function to generate the graph from the input text
        net, enhanced_docs = asyncio.run(KG.generate_knowledge_graph(int(start), int(end)))
        st.success("Knowledge graph generated successfully!")
        
        # Save the graph to an HTML file
        output_file = "knowledge_graph.html"
        net.save_graph(output_file) 

        # Open the HTML file and store it in session state
        HtmlFile = open(output_file, 'r', encoding='utf-8')
        st.session_state.graph_html = HtmlFile.read()
        
        # Store enhanced docs in session state for querying
        st.session_state.enhanced_docs = enhanced_docs
        st.session_state.graph_generated = True

# Display the graph from session state if it exists
if st.session_state.graph_html:
    components.html(st.session_state.graph_html, height=1000)

# Query Interface - Only show if graph has been generated
if st.session_state.graph_generated:
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Query Interface")
    
    query_engine = QueryEngine()
    # Query input
    query_text = st.sidebar.text_input("Enter the news:")
    query_date = st.sidebar.date_input("Published on:")
    # Query options
    top_k = st.sidebar.slider("Number of results:", min_value=1, max_value=10, value=3)
    use_tags = st.sidebar.checkbox("Use tag-based search", value=True)
    
    if st.sidebar.button("Search", type="secondary") and query_text:
        with st.spinner("Searching for similar content..."):
            try:
                # Perform the query
                results, query_tags = asyncio.run(
                    query_engine.query_with_semantic_search(
                        query_text, 
                        st.session_state.enhanced_docs, 
                        top_k=top_k, 
                        use_tags=use_tags
                    )
                )
                
                # Store results in session state to persist across reruns
                st.session_state.search_results = results
                st.session_state.query_tags = query_tags
                st.session_state.query_text = query_text
                
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
    
    # Display search results if they exist in session state
    if 'search_results' in st.session_state and st.session_state.search_results:
        results = st.session_state.search_results
        query_tags = st.session_state.query_tags
        query_text = st.session_state.query_text
        
        # Display results in the main area
        st.header("🔍 Search Results")
        st.success(f"Found {len(results)} similar documents for: '{query_text}'")
        st.write("**Query Tags:**")
        # Display tags as chips
        tags_html = " ".join(
            [f"<span style='background-color: #ff6b6b; color: white; padding: 4px 12px; margin: 4px; border-radius: 16px; font-size: 0.9em; display: inline-block;'>{tag}</span>" 
             for tag in query_tags]
        )
        st.markdown(tags_html, unsafe_allow_html=True)
        
        for i, result in enumerate(results):
            with st.expander(f"Result {i+1} - Score: {result['similarity_score']:.3f} - {result['source']}"):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.metric("Similarity", f"{result['similarity_score']:.3f}")
                    st.write(f"**News ID:** {result['news_id']}")
                    st.write(f"**Source:** {result['source']}")
                    st.write(f"**Published on** {result['pubDate']}")
                
                with col2:
                    st.write("**Tags:**")
                    # Display tags as chips
                    tags_html1 = " ".join(
                        [f"<span style='background-color: #ff6b6b; color: white; padding: 4px 12px; margin: 4px; border-radius: 16px; font-size: 0.9em; display: inline-block;'>{tag}</span>" 
                         for tag in result['tags'][:]]
                    )
                    st.markdown(tags_html1, unsafe_allow_html=True)
                
                st.divider()
                st.write("**Content Preview:**")
                st.write(result['text'])
                
                # Show extracted entities if available
                if 'document' in result and hasattr(result['document'], 'graph_document'):
                    with st.expander("View Extracted Entities"):
                        entities = []
                        for node in result['document']['graph_document'].nodes:
                            entities.append(f"{node.id} ({node.type})")
                        
                        if entities:
                            st.write("**Key Entities:**")
                            for entity in entities[:8]:
                                st.write(f"• {entity}")
                            if len(entities) > 8:
                                st.caption(f"+ {len(entities) - 8} more entities")
                        else:
                            st.info("No entities extracted")
    
    # Add some instructions
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **Query Tips:**
        - Use natural language queries
        - Enable tag-based search for better concept matching
        - Higher similarity scores indicate better matches
        """
    )

else:
    # Show instructions if no graph generated yet
    st.sidebar.info("👆 Generate a knowledge graph first to enable querying")