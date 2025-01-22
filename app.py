import streamlit as st

st.title("STREAMLIT CLASS")

st.header("PRACTICE CLASS")

st.subheader("CLASS ONE")

st.success("Success")

st.info("Info")

st.warning("Warning")

st.error("Error")

exp = ZeroDivisionError("Trying to divide by Zero")

st.exception(exp)

from PIL import Image

img = Image.open("C:\\Users\\temmy\\Desktop\\altar pics\\ikeja side a.jpg")

st.image(img)

st.checkbox("Yes/No")
st.checkbox("Blue")
st.checkbox("Black")
status = st.radio("Select Gender: ", ('Male', 'Female'))

if (status == 'Male'):

	st.success("Male")
else:
	st.success("Female")
	
hobby = st.multiselect("Hobbies: ",['Dancing', 'Reading', 'Sports'])

st.write("You selected", len("hobbies"), 'hobbies')

long_description = st.text_area(
    label="Enter a long description:",  # The label for the input field
    placeholder="Write your detailed description here..."
)

# Display the entered input
if long_description:
    st.write("Your description:")
    st.write(long_description)
	
level = st.slider("Select the level", 1, 5)

st.text('Selected: {}'.format(level))

