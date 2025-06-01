import streamlit as st
from chain import generate_lullaby

def app():
    st.set_page_config(page_title='📖 Lullaby Generator', layout='centered')
    
    st.title('📖 Lullaby Generator')
    
    st.header('What lullaby do you want to generate?')
    
    # Topic
    topic = st.text_input(label='Topic')
    
    # Genre
    genre = st.text_input(label='Genre')
    
    # Language
    language = st.text_input(label='Language')
    
    submit_button = st.button(label='Generate')
    
    if submit_button and topic and genre and language:
        with st.spinner('Generating lullaby...'):
            response = generate_lullaby(topic, genre, language)
            
            if response['story'] and response['translation']:
                st.success('Lullaby generated!')
                
                with st.expander('English Version:'):
                    st.write(response['story'])
                
                with st.expander(f'{language} Version:'):
                    st.write(response['translation'])
            else:   
                st.error('Failed to generate lullaby')

if __name__ == '__main__':
    app()