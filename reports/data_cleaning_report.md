# Data Cleaning Report

**Generated:** 2026-01-09 19:56:40

## Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total rows | 420,240 | 167,954 | -252,286 (60.0%) |
| Unique photos | 168,097 | 167,954 | - |
| Unique users | 5,158 | 5,145 | - |

## Cleaning Steps

1. **Remove corrupted dates:** 420,240 → 419,826 (removed 414)
2. **Remove duplicates:** 419,826 → 167,954 (removed 251,872)
3. **Lyon bbox filter:** 167,954 → 167,954 (removed 0)

## Date Range Comparison

| Component | Before | After |
|-----------|--------|-------|
| Year | 1 – 2238 | 1991 – 2019 |
| Month | 1 – 2011 | 1 – 12 |

## Conclusion

- ✅ All date values now valid
- ✅ No duplicate photos
- ✅ All coordinates in Lyon bbox
- ⚠️ 41,977 photos (25%) have no tags
- ⚠️ 15,776 photos (9%) have no title
