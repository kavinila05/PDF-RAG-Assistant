import streamlit as st

from rag import (
    process_pdf,
    store_chunks,
    answer_question,
    get_document_count,
    clear_database
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# 2. SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "processed_files" not in st.session_state:

    st.session_state.processed_files = set()


# ============================================================
# 3. TITLE
# ============================================================

st.title("📚 Document RAG Assistant")

st.write(
    "Upload one or more PDFs and ask questions "
    "about their contents."
)


# ============================================================
# 4. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.info(
            f"{len(uploaded_files)} PDF(s) selected."
        )

    st.divider()

    st.header("Database")

    stored_chunks = get_document_count()

    st.metric(
        "Stored chunks",
        stored_chunks
    )

    st.divider()

    if st.button(
        "🗑️ Clear database",
        use_container_width=True
    ):

        clear_database()

        st.session_state.messages = []

        st.session_state.processed_files = set()

        st.success(
            "Database cleared!"
        )

        st.rerun()


# ============================================================
# 5. PROCESS PDFS
# ============================================================

if uploaded_files:

    st.subheader("Selected documents")

    for uploaded_file in uploaded_files:

        st.write(
            f"📄 {uploaded_file.name}"
        )

    st.write("")

    if st.button(
        "⚙️ Process uploaded PDFs",
        use_container_width=True
    ):

        total_chunks = 0

        progress = st.progress(0)

        status = st.empty()

        for index, uploaded_file in enumerate(
            uploaded_files
        ):

            filename = uploaded_file.name

            # -----------------------------------------------
            # Skip files already processed during this session
            # -----------------------------------------------

            if filename in st.session_state.processed_files:

                status.info(
                    f"Skipping `{filename}` "
                    "— already processed."
                )

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

                continue

            # -----------------------------------------------
            # Process PDF
            # -----------------------------------------------

            status.write(
                f"Reading `{filename}`..."
            )

            try:

                chunks = process_pdf(
                    uploaded_file
                )

                number_of_chunks = store_chunks(
                    chunks,
                    filename
                )

                total_chunks += number_of_chunks

                st.session_state.processed_files.add(
                    filename
                )

            except ValueError as error:

                st.error(
                    f"Could not process "
                    f"`{filename}`: {error}"
                )

            except Exception as error:

                st.error(
                    f"Unexpected error while "
                    f"processing `{filename}`: "
                    f"{error}"
                )

            progress.progress(
                (index + 1) / len(uploaded_files)
            )

        status.success(
            f"Processing complete! "
            f"Added {total_chunks} chunks."
        )


# ============================================================
# 6. CHAT HISTORY DISPLAY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    content = message["content"]

    with st.chat_message(role):

        st.markdown(content)

        # -----------------------------------------------
        # Display sources for assistant messages
        # -----------------------------------------------

        if role == "assistant":

            sources = message.get(
                "sources",
                []
            )

            if sources:

                with st.expander(
                    "📚 Sources"
                ):

                    unique_sources = []

                    for source in sources:

                        if source not in unique_sources:

                            unique_sources.append(
                                source
                            )

                    for source in unique_sources:

                        st.write(
                            f"📄 **{source['source']}** "
                            f"— Page {source['page']}"
                        )


# ============================================================
# 7. CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your documents..."
)


# ============================================================
# 8. PROCESS QUESTION
# ============================================================

if question:

    # -----------------------------------------------
    # Check whether documents exist
    # -----------------------------------------------

    if get_document_count() == 0:

        st.warning(
            "Please upload and process at least "
            "one PDF before asking a question."
        )

        st.stop()


    # -----------------------------------------------
    # Add user question to chat history
    # -----------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    # -----------------------------------------------
    # Display user question
    # -----------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------------
    # Get previous conversation
    # -----------------------------------------------

    chat_history = (
        st.session_state.messages[:-1]
    )


    # -----------------------------------------------
    # Generate answer
    # -----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔍 Searching your documents..."
        ):

            answer, sources = answer_question(

                question,

                n_results=5,

                chat_history=chat_history

            )


        # -------------------------------------------
        # Display answer
        # -------------------------------------------

        st.markdown(answer)


        # -------------------------------------------
        # Display sources
        # -------------------------------------------

        if sources:

            with st.expander(
                "📚 Sources"
            ):

                unique_sources = []

                for source in sources:

                    if source not in unique_sources:

                        unique_sources.append(
                            source
                        )

                for source in unique_sources:

                    st.write(
                        f"📄 **{source['source']}** "
                        f"— Page {source['page']}"
                    )


    # -----------------------------------------------
    # Save assistant response
    # -----------------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "sources": sources

    })