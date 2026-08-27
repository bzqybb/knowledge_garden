param(
    [Parameter(Mandatory=$true)]
    [string]$PdfPath,
    [int]$StartPage = 1,
    [int]$EndPage = 0,
    [int]$TargetWidth = 1800
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime

$null = [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
$null = [Windows.Data.Pdf.PdfDocument,Windows.Data.Pdf,ContentType=WindowsRuntime]
$null = [Windows.Data.Pdf.PdfPageRenderOptions,Windows.Data.Pdf,ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream,Windows.Storage.Streams,ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder,Windows.Graphics.Imaging,ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap,Windows.Graphics.Imaging,ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult,Windows.Foundation,ContentType=WindowsRuntime]

$taskAsyncResultMethod = [System.WindowsRuntimeSystemExtensions].GetMethods() |
    Where-Object {
        $_.Name -eq "AsTask" -and $_.IsGenericMethod -and
        $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    } | Select-Object -First 1
$taskAsyncActionMethod = [System.WindowsRuntimeSystemExtensions].GetMethods() |
    Where-Object {
        $_.Name -eq "AsTask" -and -not $_.IsGenericMethod -and
        $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq "IAsyncAction"
    } | Select-Object -First 1

function Wait-WinRtResult {
    param([object]$Operation, [Type]$ResultType)
    $task = $taskAsyncResultMethod.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}

function Wait-WinRtAction {
    param([object]$Operation)
    $task = $taskAsyncActionMethod.Invoke($null, @($Operation))
    $task.Wait()
}

$resolvedPdf = (Resolve-Path -LiteralPath $PdfPath -ErrorAction Stop).Path
$file = Wait-WinRtResult (
    [Windows.Storage.StorageFile]::GetFileFromPathAsync($resolvedPdf)
) ([Windows.Storage.StorageFile])
$document = Wait-WinRtResult (
    [Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)
) ([Windows.Data.Pdf.PdfDocument])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if (-not $engine) {
    throw "Windows OCR is unavailable for the installed user languages."
}

$first = [Math]::Max(1, $StartPage)
$last = if ($EndPage -gt 0) {
    [Math]::Min([int]$document.PageCount, $EndPage)
} else {
    [int]$document.PageCount
}

for ($pageNumber = $first; $pageNumber -le $last; $pageNumber++) {
    $page = $null
    $stream = $null
    $bitmap = $null
    try {
        $page = $document.GetPage([uint32]($pageNumber - 1))
        $stream = [Windows.Storage.Streams.InMemoryRandomAccessStream]::new()
        $options = [Windows.Data.Pdf.PdfPageRenderOptions]::new()
        $scale = [Math]::Min(3.0, [Math]::Max(1.0, $TargetWidth / $page.Size.Width))
        $options.DestinationWidth = [uint32][Math]::Round($page.Size.Width * $scale)
        $options.DestinationHeight = [uint32][Math]::Round($page.Size.Height * $scale)
        Wait-WinRtAction ($page.RenderToStreamAsync($stream, $options))
        $stream.Seek(0)
        $decoder = Wait-WinRtResult (
            [Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)
        ) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap = Wait-WinRtResult (
            $decoder.GetSoftwareBitmapAsync()
        ) ([Windows.Graphics.Imaging.SoftwareBitmap])
        $result = Wait-WinRtResult (
            $engine.RecognizeAsync($bitmap)
        ) ([Windows.Media.Ocr.OcrResult])
        $lines = @($result.Lines | ForEach-Object { $_.Text })
        $payload = [ordered]@{
            page = $pageNumber
            page_count = [int]$document.PageCount
            language = $engine.RecognizerLanguage.LanguageTag
            text = [string]($lines -join "`n")
        }
        [Console]::Out.WriteLine(($payload | ConvertTo-Json -Compress -Depth 3))
        [Console]::Out.Flush()
    }
    catch {
        $payload = [ordered]@{
            page = $pageNumber
            page_count = [int]$document.PageCount
            error = [string]$_.Exception.Message
        }
        [Console]::Out.WriteLine(($payload | ConvertTo-Json -Compress -Depth 3))
        [Console]::Out.Flush()
    }
    finally {
        if ($bitmap) { $bitmap.Dispose() }
        if ($stream) { $stream.Dispose() }
        if ($page) { $page.Dispose() }
    }
}
