@echo off
echo ========================================
echo Running Grid Search for All Classification Types
echo ========================================
echo.

echo [1/3] Running 6-way classification grid search...
python grid_search.py --classification-type 6_way --quick-search --auto-confirm --save-dir grid_quicka6
if %errorlevel% neq 0 (
    echo ERROR: 6-way grid search failed!
    pause
    exit /b %errorlevel%
)
echo.
echo [1/3] 6-way classification complete!
echo.

echo [2/3] Running 3-way classification grid search...
python grid_search.py --classification-type 3_way --quick-search --auto-confirm --save-dir grid_quicka3
if %errorlevel% neq 0 (
    echo ERROR: 3-way grid search failed!
    pause
    exit /b %errorlevel%
)
echo.
echo [2/3] 3-way classification complete!
echo.

echo [3/3] Running 2-way classification grid search...
python grid_search.py --classification-type 2_way --quick-search --auto-confirm --save-dir grid_quicka2
if %errorlevel% neq 0 (
    echo ERROR: 2-way grid search failed!
    pause
    exit /b %errorlevel%
)
echo.
echo [3/3] 2-way classification complete!
echo.

echo ========================================
echo All grid searches completed successfully!
echo ========================================
echo Results saved to:
echo   - grid_quicka6/
echo   - grid_quicka3/
echo   - grid_quicka2/
echo ========================================
pause
