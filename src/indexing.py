import hashlib
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import defaultdict
import uuid
from store import get_vector_store,ensure_collection

class DocumentIndexer:
    def generate_document_id(self,path:Path)->str:
        raw=f"{path.name}:{path.stat().st_size}"

        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    
    def generate_chunk_id(self,doc_id:str,page:int,idex:int)->str:
        return f"{doc_id}:{page}:{idex}"

    def load_pdf(self,path:Path):
        pages=PyPDFLoader(str(path)).load()
        document_id=self.generate_document_id(path)
        for page in pages:
            page_number=(int(page.metadata.get("page",0))+1)

            page.metadata={
                "document_id": document_id,
                "filename": path.name,
                "source": str(path.resolve()),
                "page": page_number,
                "section": page.metadata.get("section")
            }

        return pages

    def create_split(self,chunk_size:int,chunk_overlap:int):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,          # Kích thước mỗi chunk (số ký tự)
            chunk_overlap=chunk_overlap,        # Overlap giữa các chunk
            length_function=len,     # Hàm tính độ dài
            separators=["\n\n", "\n", ". ", " ", ""],  # Thứ tự ưu tiên cắt
            keep_separator=False # không dữ separator số cuối
            )

    def build_chunks(self,pdf_path,chunk_size,chunk_overlap,chunker=None):
        #Load page
        page_docs=[]
        page_docs.extend(self.load_pdf(pdf_path))

        # tạo split
        splitter=(chunker or self.create_split(chunk_size,chunk_overlap))
        chunks=splitter.split_documents(page_docs)

        per_doc_counter=defaultdict(int)

        #normalize metadata
        for chunk in chunks:
            document_id=chunk.metadata["document_id"]
            index=per_doc_counter[document_id]

            per_doc_counter[
                document_id
            ] += 1

            chunk.metadata = {
                "document_id": document_id,
                "filename": chunk.metadata[
                    "filename"
                ],
                "source": chunk.metadata[
                    "source"
                ],
                "page": chunk.metadata[
                    "page"
                ],
                "section": chunk.metadata.get(
                    "section"
                ),

                "chunk_id":
                    self.generate_chunk_id(
                        document_id,
                        chunk.metadata["page"],
                        index
                    )
            }

        return chunks

    def index_chunk(self,chunks,collection_name=None):
        if not chunks:
            return 0
        ids=[str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"]))  for c in chunks]
        get_vector_store(collection_name=collection_name).add_documents(chunks,ids=ids)

        return len(chunks)

    def ingest(self,recreate=False, collection_name=None, chunker=None, chunk_size=None, chunk_overlap=None):
        pdfs = discover_pdfs()                    # Tìm tất cả file PDF trong thư mục
        ensure_collection(recreate=recreate,      # Đảm bảo collection tồn tại (xóa tạo lại nếu recreate=True)
                      collection_name=collection_name)
        chunks = self.build_chunks(pdfs,               # Cắt nhỏ PDF thành các chunks
                          chunker=chunker, 
                          chunk_size=chunk_size, 
                          chunk_overlap=chunk_overlap)
        return self.index_chunks(chunks, collection_name) 


    def save_and_ingest_pdf(self,file_bytes, filename):
        safe_name=Path(filename).name # lấy file và bỏ đường dẫn 
        dest=settings.data_dir/safe_name #đường dẫn đích
        settings.data_dir.mkdir(parents=True,exist_ok=True) # tạo thư mục nếu chauw có
        dest.write_bytles(file_bytes) #ghi file xuống disk
        ensure_collection(recreate=False)         # Đảm bảo collection tồn tại (không xóa cũ)
        chunks = self.build_chunks([dest])             # Chỉ xử lý file vừa lưu
        return {
            "filename": safe_name, 
            "chunks_indexed": self.index_chunks(chunks)  # Index và trả về kết quả
        }

        

