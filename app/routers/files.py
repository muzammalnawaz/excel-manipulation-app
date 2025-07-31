import os
import json
from typing import List
from fastapi import APIRouter, Request, Form, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse

from ..dependencies import verify_credentials, get_file_service
from ..services.file_service import FileService
from ..services.processors import ProcessorFactory
from ..models.user import UserInfo
from ..models.file_session import FileStatus
from ..core.config import TOOLS_CONFIG
from ..core.exceptions import FileNotFoundError, FileExpiredError, InvalidFileTypeError
from ..utils.sap_reader import get_column_info, read_sap_bi_excel

router = APIRouter(tags=["files"])

@router.post("/upload/{tool_id}")
async def upload_file(
    tool_id: str,
    request: Request,
    user_info: UserInfo = Depends(verify_credentials),
    file_service: FileService = Depends(get_file_service)
):
    """Upload file(s) for processing"""
    print(f"\n=== UPLOAD DEBUG START ===")
    print(f"Tool ID: {tool_id}")
    print(f"User: {user_info.username}")
    
    if tool_id not in TOOLS_CONFIG:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    try:
        form = await request.form()
        print(f"Form keys: {list(form.keys())}")
        
        if tool_id == "stock_distribution":
            # Handle dual file upload for stock distribution
            warehouse_file = form.get("warehouse_file")
            sales_file = form.get("sales_file")
            
            if not warehouse_file or not sales_file:
                raise HTTPException(status_code=400, detail="Both warehouse_file and sales_file are required")
            
            if not hasattr(warehouse_file, 'filename') or not hasattr(sales_file, 'filename'):
                raise HTTPException(status_code=400, detail="Invalid file uploads")
            
            print(f"Files received - Warehouse: {warehouse_file.filename}, Sales: {sales_file.filename}")
            
            # Create file session
            file_id = file_service.create_file_session(tool_id, user_info.username)
            
            try:
                # Save files
                warehouse_content = await warehouse_file.read()
                warehouse_path = file_service.save_uploaded_file(
                    warehouse_content, warehouse_file.filename, file_id, "warehouse"
                )
                
                sales_content = await sales_file.read()
                sales_path = file_service.save_uploaded_file(
                    sales_content, sales_file.filename, file_id, "sales"
                )
                
                # Update session with file information
                file_service.update_file_session(
                    file_id,
                    warehouse_path=warehouse_path,
                    sales_path=sales_path,
                    warehouse_filename=warehouse_file.filename,
                    sales_filename=sales_file.filename
                )
                
                # Validate files can be read
                warehouse_df = read_sap_bi_excel(warehouse_path, "warehouse")
                sales_df = read_sap_bi_excel(sales_path, "sales")
                
                if warehouse_df.empty:
                    raise Exception("No valid data found in warehouse file")
                
                if sales_df.empty:
                    raise Exception("No valid data found in sales file")
                
                print(f"SUCCESS: Files uploaded with ID {file_id}")
                print(f"=== UPLOAD DEBUG END ===\n")
                
                return {
                    "file_id": file_id,
                    "warehouse_filename": warehouse_file.filename,
                    "sales_filename": sales_file.filename,
                    "message": "Files uploaded successfully. Ready for processing."
                }
            
            except Exception as e:
                # Clean up files on error
                try:
                    session = file_service.get_file_session(file_id)
                    for path in [session.warehouse_path, session.sales_path]:
                        if path and os.path.exists(path):
                            os.remove(path)
                except:
                    pass
                raise HTTPException(status_code=400, detail=f"Error processing files: {str(e)}")
        
        else:
            # Handle single file upload
            file = form.get("file")
            
            if not file:
                raise HTTPException(status_code=400, detail="File is required")
            
            if not hasattr(file, 'filename'):
                raise HTTPException(status_code=400, detail="Invalid file upload")
            
            # Create file session
            file_id = file_service.create_file_session(tool_id, user_info.username)
            
            try:
                # Save file
                content = await file.read()
                file_path = file_service.save_uploaded_file(content, file.filename, file_id)
                
                # Read and analyze file
                df = read_sap_bi_excel(file_path, "general")
                
                if df.empty:
                    raise Exception("No valid data found in Excel file")
                
                columns = df.columns.tolist()
                column_info = get_column_info(df) if tool_id == "data_summarizer" else {}
                
                # Update session with file information
                file_service.update_file_session(
                    file_id,
                    original_path=file_path,
                    original_filename=file.filename,
                    columns=columns,
                    column_info=column_info
                )
                
                response_data = {
                    "file_id": file_id,
                    "columns": columns,
                    "filename": file.filename
                }
                
                if tool_id == "data_summarizer":
                    response_data["column_info"] = column_info
                
                return response_data
            
            except Exception as e:
                # Clean up file on error
                try:
                    session = file_service.get_file_session(file_id)
                    if session.original_path and os.path.exists(session.original_path):
                        os.remove(session.original_path)
                except:
                    pass
                raise HTTPException(status_code=400, detail=f"Error reading Excel file: {str(e)}")
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in upload: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.post("/process/{file_id}")
async def process_file(
    file_id: str,
    request: Request,
    user_info: UserInfo = Depends(verify_credentials),
    file_service: FileService = Depends(get_file_service)
):
    """Process uploaded file(s)"""
    try:
        session = file_service.get_file_session(file_id)
        
        # Verify user owns this session
        if session.user != user_info.username:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get processor configuration
        tool_config = TOOLS_CONFIG[session.tool_id]
        processor_name = tool_config["processor"]
        
        # Create processor
        processor = ProcessorFactory.create_processor(processor_name, file_id)
        
        # Process based on tool type
        if processor_name == "process_stock_distribution":
            processed_path = processor.process(session.warehouse_path, session.sales_path)
            
        elif processor_name == "split_excel_by_column":
            form = await request.form()
            column_name = form.get("column_name")
            
            if not column_name or column_name not in session.columns:
                raise HTTPException(status_code=400, detail="Valid column name required")
            
            processed_path = processor.process(session.original_path, column_name, session.columns)
            
        elif processor_name == "summarize_data":
            form = await request.form()
            groupby_columns = form.get("groupby_columns")
            numeric_columns = form.get("numeric_columns")
            operation = form.get("operation")
            
            if not groupby_columns or not numeric_columns or not operation:
                raise HTTPException(status_code=400, detail="All fields are required for summarization")
            
            try:
                groupby_cols = json.loads(groupby_columns)
                numeric_cols = json.loads(numeric_columns)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid columns format")
            
            processed_path = processor.process(
                session.original_path, 
                groupby_cols, 
                numeric_cols,
                operation, 
                session.columns
            )
            
        else:
            raise HTTPException(status_code=400, detail="Unknown processor")
        
        # Update session
        file_service.update_file_session(
            file_id,
            processed=True,
            processed_path=processed_path
        )
        
        return {"message": "File processed successfully", "download_ready": True}
    
    except (FileNotFoundError, FileExpiredError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/download/{file_id}")
async def download_file(
    file_id: str, 
    user_info: UserInfo = Depends(verify_credentials),
    file_service: FileService = Depends(get_file_service)
):
    """Download processed file"""
    try:
        session = file_service.get_file_session(file_id)
        
        # Verify user owns this session
        if session.user != user_info.username:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not session.processed:
            raise HTTPException(status_code=400, detail="File not processed yet")
        
        processed_path = session.processed_path
        if not os.path.exists(processed_path):
            raise HTTPException(status_code=404, detail="Processed file not found")
        
        # Generate appropriate filename
        if session.tool_id == "stock_distribution":
            download_filename = "Stock_Distribution_Analysis.xlsx"
        elif session.tool_id == "data_summarizer":
            original_name = session.original_filename
            name_without_ext = os.path.splitext(original_name)[0]
            download_filename = f"{name_without_ext}_summarized.xlsx"
        else:
            original_name = session.original_filename
            name_without_ext = os.path.splitext(original_name)[0]
            download_filename = f"{name_without_ext}_processed.xlsx"
        
        return FileResponse(
            processed_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=download_filename
        )
    
    except (FileNotFoundError, FileExpiredError) as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/file_status/{file_id}")
async def file_status(
    file_id: str, 
    user_info: UserInfo = Depends(verify_credentials),
    file_service: FileService = Depends(get_file_service)
) -> FileStatus:
    """Get file status information"""
    try:
        session = file_service.get_file_session(file_id)
        
        # Verify user owns this session
        if session.user != user_info.username:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return file_service.get_file_status(file_id)
    
    except (FileNotFoundError, FileExpiredError) as e:
        raise HTTPException(status_code=404, detail=str(e))