# pipelines/data_monitoring/nodes.py
import pandas as pd
from typing import Dict
from evidently.test_suite import TestSuite
from evidently.tests import (
    # TestMostCommonValueShare,
    TestColumnShareOfMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfDuplicatedColumns,
    TestNumberOfDuplicatedRows
)
from evidently.test_preset import DataDriftTestPreset
import logging

logger = logging.getLogger(__name__)

def run_data_validation(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    monitoring_params: dict
) -> Dict:
    """Run validation tests with improved error handling."""
    
    # Log dataset statistics
    
    # Create test suite with more lenient thresholds
    test_suite = TestSuite(tests=[
        TestColumnShareOfMissingValues(
            column_name="Text",
        ),
        # TestMostCommonValueShare(
        #     column_name="Text",
        # ),
        TestNumberOfConstantColumns(),
        TestNumberOfDuplicatedColumns(),
        TestNumberOfDuplicatedRows(),
        DataDriftTestPreset(
            columns=["Score"],
            cat_stattest=monitoring_params["drift"]["cat_stattest"],
            cat_stattest_threshold=monitoring_params["drift"]["cat_stattest_threshold"],
            drift_share=monitoring_params["drift"]["drift_share"],
        ),
    ])
    
    # Run tests
    test_suite.run(reference_data=reference_data, current_data=current_data)
    test_results = test_suite.as_dict()
    
    # Enhanced logging
    logger.info("\nDetailed Test Results:")
    for test in test_results["tests"]:
        logger.info(f"\nTest: {test['name']}")
        logger.info(f"Status: {test['status']}")
        
        # Log the actual values and thresholds
        if 'parameters' in test:
            logger.info("Test Parameters:")
            for key, value in test['parameters'].items():
                logger.info(f"  {key}: {value}")
        
        if 'description' in test:
            logger.info(f"Description: {test['description']}")
    
    # Save HTML report
    report_path = f"data/09_monitoring/test_results.html"
    test_suite.save_html(report_path)
    logger.info(f"\nDetailed report saved to: {report_path}")
    
    # Check for failures with more context
    failed_tests = [
        test for test in test_results["tests"]
        if test['status'] == 'FAIL' and 'drift' not in test['name'].lower()
    ]
    
    if failed_tests:
        error_msg = "\nData validation failed:\n"
        for test in failed_tests:
            error_msg += f"\nTest: {test['name']}\n"
            error_msg += f"Status: {test['status']}\n"
            if 'description' in test:
                error_msg += f"Description: {test['description']}\n"
            if 'parameters' in test:
                error_msg += f"Parameters: {test['parameters']}\n"
            
            # Add context based on test type
            if "quality" in test['name'].lower():
                error_msg += "Action Required: Check for duplicate or malformed data\n"
        
        raise ValueError(error_msg)
    
    return test_results