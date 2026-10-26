# Register the Task Scheduler entries that start Stoa's jobs on time.
# Times are New York time; the Dell's clock is set to Eastern, so daylight
# saving time is handled by Windows. Weekdays only; Stoa skips NYSE holidays.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install-dispatch.ps1

$script = Join-Path $PSScriptRoot "dispatch.ps1"
$days = "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"

# The sweep starts at 8:57 and holds until 9:00, so the picks go out at 9:00
# with the night's news complete. EXECUTE opens at 9:45 and does not wait.
$jobs = @(
    @{ Name = "Stoa sweep"; Job = "sweep"; At = "8:57am" },
    @{ Name = "Stoa execute"; Job = "execute"; At = "9:45am" },
    @{ Name = "Stoa decide"; Job = "decide"; At = "5:15pm" },
    @{ Name = "Stoa analyst"; Job = "analyst"; At = "7:45pm" }
)

foreach ($j in $jobs) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`" -Job $($j.Job)"
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At $j.At
    $settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
    Register-ScheduledTask -TaskName $j.Name -Action $action -Trigger $trigger `
        -Settings $settings -Force | Out-Null
    Write-Output "$($j.Name): weekdays at $($j.At)"
}
