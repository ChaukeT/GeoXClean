# Resize GPU Timeout - Debugging Guide

## New Debug Logging Added

I've added comprehensive debug logging to track exactly what's happening during resize. Here's what to look for in the logs:

---

## 1. Resize Event Tracking

### What to Look For:
```
[RESIZE #1] 800x600 → 1024x768 (Δ0ms)
[RESIZE #2] 1024x768 → 1200x900 (Δ18ms)
[RESIZE #3] 1200x900 → 1400x1050 (Δ16ms)
```

**What This Tells You:**
- **Resize #X**: Sequential resize counter
- **Δ18ms**: Time since last resize event

**Look For Problems:**
- **Rapid resizes** (Δ < 50ms): Normal during window drag
- **Δ > 1000ms**: New resize after pause (expected)
- **Δ very small (< 10ms) repeatedly**: Possible loop!

---

## 2. Resize Loop Detection

### What to Look For:
```
[RESIZE LOOP DETECTED] 20 resizes in rapid succession!
Stack trace:
  File "viewer_widget.py", line 215, in resizeEvent
  File "...Qt framework..."
```

**What This Means:**
- More than 10 resizes in under 100ms each
- Likely a loop where resize triggers another resize
- **Stack trace shows WHO is calling resizeEvent** (Qt, VTK, or your code)

---

## 3. Resize Completion

### What to Look For:
```
[RESIZE COMPLETE] Called at t=1707859537.095, total resize count: 3
[RESIZE COMPLETE] has_large_model=True, total_cells=251,100
```

**What This Tells You:**
- **t=timestamp**: When debounce completed
- **total resize count**: How many resizes since app started
- **total_cells**: Combined complexity of all layers

**Expected Behavior:**
- Should appear ~500ms after last resize event (for large models)
- Should appear ~200ms after last resize event (for normal models)

---

## 4. Render Strategy

### For Normal Models (<30k cells):
```
[RESIZE COMPLETE] Re-enabling render for normal model
[RESIZE COMPLETE] Render completed
```
**Expected**: Immediate re-enable and render

### For Large Models (30k-200k cells):
```
[RESIZE COMPLETE] Large model (150,000 cells) - delaying render re-enable by 2 seconds
[DELAYED ENABLE] Called at t=1707859539.095
[DELAYED ENABLE] Re-enabling VTK rendering
[DELAYED ENABLE] VTK rendering re-enabled successfully
```
**Expected**: 2-second delay before re-enabling

### For EXTREME Models (>200k cells):
```
[RESIZE COMPLETE] EXTREME model (251,100 cells) - keeping render DISABLED permanently
[RESIZE COMPLETE] User must manually rotate/interact to trigger render
```
**Expected**: Rendering stays disabled! User must interact to re-enable.

---

## How to Use This Debug Info

### Test 1: Normal Resize
1. Start app with small model (<30k cells)
2. Resize window (drag edge or maximize)
3. **Check logs for:**
   - Resize events with reasonable Δ times (10-100ms)
   - Resize complete message
   - "Re-enabling render for normal model"
4. **Expected**: Smooth resize, no timeout

### Test 2: Large Model Resize
1. Load drillholes + SGSIM grid (~100k-200k total cells)
2. Resize window
3. **Check logs for:**
   - Resize events
   - "Large model (X cells) - delaying render re-enable"
   - After 2 seconds: "DELAYED ENABLE" messages
4. **Expected**: Delayed render, no timeout

### Test 3: Extreme Model (Your Current Case)
1. Current setup: 251,100 cells
2. Resize window
3. **Check logs for:**
   - Resize events (count them!)
   - "EXTREME model (251,100 cells) - keeping render DISABLED permanently"
   - NO delayed enable messages
   - NO "Re-enabled rendering" message
4. **Expected**:
   - Render stays disabled after resize
   - User must manually rotate/pan to trigger render
   - Should NOT timeout (render never happens!)

---

## Identifying the Issue

### Scenario A: Resize Loop
**Logs show:**
```
[RESIZE LOOP DETECTED] 50 resizes in rapid succession!
```
**Cause**: Something in the code is triggering resize events recursively
**Solution**: Check the stack trace to see what's calling resizeEvent

### Scenario B: Delayed Render Triggers Resize
**Logs show:**
```
[DELAYED ENABLE] Re-enabling VTK rendering
[RESIZE #25] ... (immediately after)
```
**Cause**: Enabling rendering triggers a window update which fires resize
**Solution**: Don't re-enable rendering for extreme models (already implemented)

### Scenario C: Grid Too Large
**Logs show:**
```
[RESIZE COMPLETE] EXTREME model (251,100 cells)
```
**Cause**: Your SGSIM grid has 251,100 cells - that's too many!
**Solution**:
- Create SGSIM on coarser grid (increase cell size)
- Target <100k cells total
- Use composite spacing (5-10m) not sample spacing (1-2m)

---

## Next Steps

1. **Run the app** with current setup
2. **Resize the window** (both shrink and grow)
3. **Copy the logs** showing:
   - All `[RESIZE #...]` lines
   - All `[RESIZE COMPLETE]` lines
   - Any `[RESIZE LOOP DETECTED]` warnings
   - Any `[DELAYED ENABLE]` lines
4. **Send me the log output** so we can see exactly what's happening

---

## Quick Fix for Your Current Issue

Your scene has **251,100 cells** - that's in the EXTREME category. The new code should:
1. Detect this during resize
2. Keep rendering **permanently disabled** after resize
3. NOT trigger any delayed enable
4. NOT render at all (avoiding GPU timeout)

**However**, the visualization will appear "frozen" after resize. To update the view:
- **Rotate** the scene (click and drag)
- **Pan** the scene (shift + drag)
- Any interaction will trigger a render

This is intentional - it prevents automatic renders that cause GPU timeout.

---

## Permanent Solution

**Reduce your SGSIM grid size:**
1. Open SGSIM panel
2. Reduce nx, ny, nz to create fewer cells
3. Target 50x50x20 = **50,000 cells** (safe)
4. Current grid is likely ~100x100x25 = **250,000 cells** (extreme)

**Or disable drillhole layer during resize:**
1. Before resizing, click "Drillholes" quick layer to hide them
2. Resize window
3. Click "Drillholes" again to show them
4. Total cells during resize: just the SGSIM grid (much less)
