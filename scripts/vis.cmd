set current_dir=%cd%
pushd "C:\UofA\Mojow\mj_code\synthetic-data-generation"
python3 -m src.tests.visualize_rock_data  src.dir=%current_dir%
popd

