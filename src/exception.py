import sys
from src.logger import logging

def get_detailed_error_message(error_messages,error_details:sys):
    _,_,er = error_details.exc_info()
    filename= er.tb_frame.f_code.co_filename
    error_message=f"Error ocuured at filename {filename} at line number {er.tb_lineno} with error message{str(error_messages)}"

    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message= get_detailed_error_message(error_message,error_details=error_detail)

    def __str__(self) -> str:
        return self.error_message
    
if __name__=="__main__":
    try:
        a=4/0
    except Exception as e:
        logging.info("Logging Tested")
        raise CustomException(e,sys)