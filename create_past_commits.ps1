# PowerShell script to create commits on past random dates
# This populates your git contribution graph

# Define random past dates (adjust the range as needed)
$dates = @(
  "2025-09-15 14:30:00",
  "2025-09-22 10:15:00",
  "2025-10-05 16:45:00",
  "2025-10-18 09:20:00",
  "2025-11-02 13:10:00",
  "2025-11-15 11:30:00",
  "2025-12-01 15:20:00",
  "2025-12-10 08:45:00",
  "2026-01-05 12:30:00",
  "2026-01-20 14:15:00"
)

# Create commits
foreach ($date in $dates) {
  $dateFormatted = (Get-Date -Date $date -Format 'yyyy-MM-dd')
  Write-Host "Creating commit for $dateFormatted..."
  
  # Add a line to activity log
  Add-Content -Path "activity_log.txt" -Value "Activity on $dateFormatted`r`n"
  
  # Stage and commit
  git add .
  git commit --date "$date" -m "Activity on $dateFormatted"
}

Write-Host "Done! Your git graph should now show activity on these dates."
Write-Host "Push to remote with: git push origin main (or your branch name)"
