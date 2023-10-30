# from pypdf import PdfReader

# reader = PdfReader("2022 Fall Midterm 1 KEY.pdf")
# number_of_pages = len(reader.pages)
# page = reader.pages[22]
# text = page.extract_text()
# print(text)
# from langchain.document_loaders import PDFMinerLoader
# loader = PDFMinerLoader("2023 Spring Midterm 1.pdf")
# data = loader.load()
# print(data) 


# import fitz

# pdffile = "2022 Fall Midterm 1.pdf"
# doc = fitz.open(pdffile)
# pageCount= doc.page_count
# page = doc.load_page(1)  # number of page
# pix = page.get_pixmap()
# output = "outfile.png"
# pix.save(output)
# doc.close()
# import pytesseract 
# from PIL import Image 
 
# # Load the image 
# img = Image.open("outfile.png") 
 
# # # Use Tesseract to extract text 
# extracted_text = pytesseract.image_to_string(img) 
 
# print(extracted_text)


filename = "MSTQns.txt"

# Open and read the file
topicList = []

with open(filename, 'r') as file:
    for line in file:
        topicList.append(line.strip())

print(topicList)
