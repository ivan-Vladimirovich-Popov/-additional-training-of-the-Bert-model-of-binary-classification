import openpyxl
import xlsxwriter
new_book1=xlsxwriter.Workbook("test_data_with_labels.xlsx")
worksheet = new_book1.add_worksheet()
book=openpyxl.open("dataset for test.xlsx",read_only=True)
sheet=book.active
count=0
worksheet.write(f"A{1}","labels")
worksheet.write(f"B{1}","text")
for row in range(1,1000):
    grade=sheet[row][0].value
    hotel_review=sheet[row][1].value
    if int(grade)<3:
        worksheet.write(f"A{1+row}",0)
        worksheet.write(f"B{1+row}",hotel_review)
    else:
        worksheet.write(f"A{1+row}",1)
        worksheet.write(f"B{1+row}",hotel_review)
    count+=1
    print(count)
new_book1.close()

