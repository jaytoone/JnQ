# Changelog

## [1.1.5] - 2024-07-12
### Added
- Internalized TokenBucket logic
- Set Bank show_header=True by default

## [1.1.4] - 2024-07-10
### Added
- Incorporated TokenBucket functionality

## [1.1.3] - 2024-07-08
### Added
- Integrated DBMS
- Separated send/server logic
### Changed
- Applied dynamic api_count adjustment (+8)

## [1.1.2] - 2024-07-05
### Changed
- Minimized unnecessary self references

## [1.1.1] - 2024-07-03
### Changed
- Enhanced input/output readability
- Added newline formatting for rows

## [1.1.0] - 2024-07-01
### Added
- Added trade completion check
- Migrated to feather format (orderId as int64)
- Integrated side_open in check_stop_loss
- Refined statusChangeTime ordering and formatting
- Implemented error handling for remove_row
- Updated get_income_info with side_position
- Enhanced websocket_client remove logic
- Improved get_price_realtime error handling
- Safeguarded orderId with set_table integration

## [1.0.0] - 2024-06-20
### Added
- Initial release
- Enforced static row values
- Prevented duplicate order_market entries
- Optimized order phase status updates
- Unified table_log concatenation with loc
- Ensured secure orderId dtype handling
