# streamlit_app.py

import streamlit as st
from infer_simple_def import predict_fn

def ml_model(seq_mut):
    return predict_fn("GDSFTHTP", seq_mut);

st.set_page_config(page_title="Character→Image", layout="wide")

colp_1, colp_2, colp_3 = st.columns([1, 1, 5])
with colp_1:
    st.title("DeltaG")
with colp_2:
    st.image("img/211051808.jpeg", width=100)

st.markdown("## Demo: EGFR oncogen protein antibody binding sequence")
col1, col2 = st.columns(2)

with col1:
    
    st.markdown("The model predicts the probability that a given **EGFR mutation** will **escape** binding by a **therapeutic antibody**.")

    st.markdown("### Try it out!")
    st.markdown("Change the Amino Acide in the below box to get the mutation escape score.")

    st.markdown("**N-end** LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFR") 

    ## EGFR antibody binding sequence
    prefill = "GDSFTHTP"

    # Make one column per character
    cols = st.columns(len(prefill))

    # Render a 1-char text_input in each, starting with the prefill
    chars = []
    for i, (col, ch) in enumerate(zip(cols, prefill)):
        c = col.text_input(
            label="",            # no label
            value=ch,            # prefill character
            max_chars=1,         # only one character
            key=f"char_{i}"
        )
        chars.append(c or "")    # make sure it's a string

    # Join them back together
    result = "".join(chars)

    # Show the combined result
    # st.markdown(f"**Binding area:** {result}")

    st.markdown("  PGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPS **C-end**")
    
with col2:

    col2_1, col2_2 = st.columns(2)
    output = ml_model(result)
    output = round(output[0] * 100, 0)
    output = round((output + 24) * -1 + 0.01, 0)

    if output > 10:
        with col2_1:
            st.markdown(
                f"""
                <div style='display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%;'>
                    <div style='text-align:center; font-size:30px'>Escape Score:</div>
                    <br>
                    <div style='text-align:center; font-size: 100px; color: #FF6347;'>{output}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2_2:
            st.image("./img/alarm.gif")
    else:
        st.markdown(
            f"""
            <div style='display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%;'>
                <div style='text-align:center; font-size:30px'>Escape Score:</div>
                <br>
                <div style='text-align:center; font-size: 100px; color: #32CD32;'>{output}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.image("./img/small_highlight.gif")

st.markdown("### How does it work?")
st.image("img/model_flow.png")
