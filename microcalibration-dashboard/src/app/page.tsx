'use client';

import { useState, useEffect, useCallback } from 'react';
import FileUpload from '@/components/FileUpload';
import MetricsOverview from '@/components/MetricsOverview';
import LossChart from '@/components/LossChart';
import ErrorDistribution from '@/components/ErrorDistribution';
import CalibrationSummary from '@/components/CalibrationSummary';
import SingleDatasetBarChart from '@/components/SingleDatasetBarChart';
import ComparisonSummary from '@/components/ComparisonSummary';
import ComparisonCharts from '@/components/ComparisonCharts';
import ComparisonQualitySummary from '@/components/ComparisonQualitySummary';
import RegressionAnalysis from '@/components/RegressionAnalysis';
import TargetConvergenceComparison from '@/components/TargetConvergenceComparison';
import DataTable from '@/components/DataTable';
import ComparisonDataTable from '@/components/ComparisonDataTable';
import SanityCheckPanel from '@/components/SanityCheckPanel';
import ValidationSummary from '@/components/ValidationSummary';
import ValidationDataTable from '@/components/ValidationDataTable';
import ValidationByAreaChart from '@/components/ValidationByAreaChart';
import ValidationScatterPlot from '@/components/ValidationScatterPlot';
import CalibrationVsSimComparison from '@/components/CalibrationVsSimComparison';
import StateRollupComparison from '@/components/StateRollupComparison';
import UnifiedResultsTable from '@/components/UnifiedResultsTable';
import { CalibrationDataPoint, ValidationDataPoint } from '@/types/calibration';
import { parseCalibrationCSV } from '@/utils/csvParser';
import { isValidationCsv, parseValidationCSV } from '@/utils/validationCsvParser';
import { getCurrentDeeplinkParams, generateShareableUrl, DeeplinkParams, HuggingFaceDeeplinkInfo } from '@/utils/deeplinks';
import { Share, CheckCircle, Upload } from 'lucide-react';

type ViewMode = 'calibration' | 'validation' | 'explorer';

export default function Dashboard() {
  const [data, setData] = useState<CalibrationDataPoint[]>([]);
  const [validationData, setValidationData] = useState<ValidationDataPoint[]>([]);
  const [filename, setFilename] = useState<string>('');
  const [validationFilename, setValidationFilename] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [showDashboard, setShowDashboard] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<ViewMode>('calibration');

  // Comparison mode state
  const [comparisonMode, setComparisonMode] = useState<boolean>(false);
  const [secondData, setSecondData] = useState<CalibrationDataPoint[]>([]);
  const [secondFilename, setSecondFilename] = useState<string>('');

  // Deeplink state
  const [deeplinkParams, setDeeplinkParams] = useState<DeeplinkParams | null>(null);
  const [isLoadingFromDeeplink, setIsLoadingFromDeeplink] = useState<boolean>(false);

  // GitHub artifact state for sharing
  const [githubArtifactInfo, setGithubArtifactInfo] = useState<DeeplinkParams | null>(null);

  // Validation drop zone state
  const [valDragOver, setValDragOver] = useState(false);

  const handleFileLoad = useCallback((content: string, name: string) => {
    try {
      if (isValidationCsv(content)) {
        const parsed = parseValidationCSV(content);
        setValidationData(parsed);
        setValidationFilename(name);
        setError('');
        if (data.length > 0) {
          setViewMode('explorer');
        } else {
          setViewMode('validation');
        }
        if (isLoadingFromDeeplink) {
          setShowDashboard(true);
        }
      } else {
        const parsedData = parseCalibrationCSV(content);
        setData(parsedData);
        setFilename(name);
        setError('');
        setComparisonMode(false);
        if (validationData.length > 0) {
          setViewMode('explorer');
        } else {
          setViewMode('calibration');
        }
        if (isLoadingFromDeeplink) {
          setShowDashboard(true);
        }
      }
    } catch (err) {
      console.error('Error parsing CSV:', err);
      setError(err instanceof Error ? err.message : 'Failed to parse CSV file');
    }
  }, [data.length, validationData.length, isLoadingFromDeeplink]);

  const handleComparisonLoad = (content1: string, filename1: string, content2: string, filename2: string) => {
    try {
      const parsedData1 = parseCalibrationCSV(content1);
      const parsedData2 = parseCalibrationCSV(content2);

      setData(parsedData1);
      setFilename(filename1);
      setSecondData(parsedData2);
      setSecondFilename(filename2);
      setComparisonMode(true);
      setViewMode('calibration');
      setError('');
      setShowDashboard(true);
    } catch (err) {
      console.error('Error parsing comparison CSV:', err);
      setError(err instanceof Error ? err.message : 'Failed to parse comparison CSV files');
      setData([]);
      setSecondData([]);
      setFilename('');
      setSecondFilename('');
      setComparisonMode(false);
      setShowDashboard(false);
    }
  };

  const handleHuggingFaceDeeplink = async (hf: HuggingFaceDeeplinkInfo) => {
    setIsLoadingFromDeeplink(true);
    const fetchHfFile = async (path: string) => {
      const revision = hf.revision || 'main';
      const url = `https://huggingface.co/${hf.repo}/resolve/${revision}/${path}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HF fetch failed: ${res.status} for ${path}`);
      return res.text();
    };

    try {
      const calContent = await fetchHfFile(hf.calPath);
      const calFilename = `${hf.calPath.split('/').pop()} (HF)`;
      handleFileLoad(calContent, calFilename);

      if (hf.valPath) {
        const valContent = await fetchHfFile(hf.valPath);
        const valFilename = `${hf.valPath.split('/').pop()} (HF)`;
        handleFileLoad(valContent, valFilename);
      }

      setShowDashboard(true);
    } catch (err) {
      console.error('HF deeplink load failed:', err);
      setError(err instanceof Error ? err.message : 'Failed to load from HuggingFace deeplink');
    } finally {
      setIsLoadingFromDeeplink(false);
    }
  };

  // Check for deeplink parameters on mount
  useEffect(() => {
    const params = getCurrentDeeplinkParams();
    if (params) {
      setDeeplinkParams(params);
      if (params.mode === 'huggingface' && params.huggingface) {
        handleHuggingFaceDeeplink(params.huggingface);
      } else {
        setIsLoadingFromDeeplink(true);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const generateShareUrl = (): string => {
    const shareParams = githubArtifactInfo || deeplinkParams;
    if (!shareParams) return window.location.href;
    return generateShareableUrl(shareParams);
  };

  const handleShare = async () => {
    try {
      const shareUrl = generateShareUrl();
      await navigator.clipboard.writeText(shareUrl);
      alert('Shareable URL copied to clipboard!');
    } catch (err) {
      console.error('Failed to copy URL:', err);
      alert('Failed to copy URL to clipboard');
    }
  };

  const handleValFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setValDragOver(false);
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      handleFileLoad(ev.target?.result as string, file.name);
    };
    reader.readAsText(file);
  };

  const handleValFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      handleFileLoad(ev.target?.result as string, file.name);
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const hasCalibration = data.length > 0;
  const hasValidation = validationData.length > 0;
  const hasBoth = hasCalibration && hasValidation;

  const availableTabs: ViewMode[] = [];
  if (hasCalibration) availableTabs.push('calibration');
  if (hasValidation) availableTabs.push('validation');
  if (hasBoth) availableTabs.push('explorer');

  useEffect(() => {
    if (showDashboard && !availableTabs.includes(viewMode)) {
      if (availableTabs.length > 0) {
        setViewMode(availableTabs[0]);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showDashboard, hasCalibration, hasValidation]);

  const loadedDescription = [
    filename && `Cal: ${filename}`,
    comparisonMode && secondFilename && `vs ${secondFilename}`,
    validationFilename && `Val: ${validationFilename}`,
  ].filter(Boolean).join(' | ');

  const resetAll = () => {
    setData([]);
    setSecondData([]);
    setValidationData([]);
    setFilename('');
    setSecondFilename('');
    setValidationFilename('');
    setComparisonMode(false);
    setError('');
    setShowDashboard(false);
    setViewMode('calibration');
    setDeeplinkParams(null);
    setIsLoadingFromDeeplink(false);
    setGithubArtifactInfo(null);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 text-gray-900">
            Calibration dashboard
          </h1>
          <p className="text-gray-600 text-lg">
            Microdata weight calibration assessment
          </p>
          {showDashboard && loadedDescription && (
            <p className="mt-1 text-sm text-blue-600">{loadedDescription}</p>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-md p-4">
            <div className="flex">
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">
                  Error loading file
                </h3>
                <div className="mt-2 text-sm text-red-700">
                  {error}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Pre-dashboard: File Upload */}
        {!showDashboard && (
          <div className="space-y-6 mb-8">
            {/* Layer 1: Calibration data */}
            <div className="bg-white rounded-lg shadow-lg overflow-hidden">
              <div className="px-6 py-4 bg-gray-800 text-white flex items-center gap-3">
                <span className={`flex items-center justify-center w-7 h-7 rounded-full text-sm font-bold ${
                  hasCalibration ? 'bg-green-500' : 'bg-gray-500'
                }`}>1</span>
                <div>
                  <h2 className="font-semibold">Calibration log</h2>
                  <p className="text-sm text-gray-300">Training loss and per-target estimates from the calibration run</p>
                </div>
                {hasCalibration && (
                  <CheckCircle size={20} className="ml-auto text-green-400" />
                )}
              </div>

              {hasCalibration ? (
                <div className="px-6 py-4 bg-green-50 border-b border-green-200">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-green-700 text-sm">
                      <CheckCircle size={16} />
                      <span className="font-medium">{filename}</span>
                      <span className="text-green-600">
                        ({data.filter((d, i, a) => a.findIndex(x => x.target_name === d.target_name) === i).length} targets,
                        {' '}{Math.max(...data.map(d => d.epoch))} epochs)
                      </span>
                    </div>
                    <button
                      onClick={() => { setData([]); setFilename(''); setComparisonMode(false); setSecondData([]); setSecondFilename(''); }}
                      className="text-xs text-gray-500 hover:text-red-600"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ) : (
                <div className="p-6">
                  <FileUpload
                    onFileLoad={handleFileLoad}
                    onCompareLoad={handleComparisonLoad}
                    onViewDashboard={() => setShowDashboard(true)}
                    deeplinkParams={deeplinkParams}
                    isLoadingFromDeeplink={isLoadingFromDeeplink}
                    onDeeplinkLoadComplete={(primary, secondary) => {
                      const params = { mode: secondary ? 'comparison' : 'single', primary, secondary } as DeeplinkParams;
                      setDeeplinkParams(params);
                      setGithubArtifactInfo(params);
                      setIsLoadingFromDeeplink(false);
                      if (primary) {
                        setShowDashboard(true);
                      }
                    }}
                    onGithubLoad={(primary, secondary) => {
                      setGithubArtifactInfo({ mode: secondary ? 'comparison' : 'single', primary, secondary });
                    }}
                  />
                </div>
              )}
            </div>

            {/* Layer 2: Validation data */}
            <div className={`bg-white rounded-lg shadow-lg overflow-hidden transition-opacity ${
              hasCalibration || hasValidation ? 'opacity-100' : 'opacity-40 pointer-events-none'
            }`}>
              <div className="px-6 py-4 bg-gray-700 text-white flex items-center gap-3">
                <span className={`flex items-center justify-center w-7 h-7 rounded-full text-sm font-bold ${
                  hasValidation ? 'bg-green-500' : 'bg-gray-500'
                }`}>2</span>
                <div>
                  <h2 className="font-semibold">Validation results
                    <span className="ml-2 text-xs font-normal text-gray-400">(optional)</span>
                  </h2>
                  <p className="text-sm text-gray-300">Sim-vs-target comparison from validate_staging</p>
                </div>
                {hasValidation && (
                  <CheckCircle size={20} className="ml-auto text-green-400" />
                )}
              </div>

              {hasValidation ? (
                <div className="px-6 py-4 bg-green-50">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-green-700 text-sm">
                      <CheckCircle size={16} />
                      <span className="font-medium">{validationFilename}</span>
                      <span className="text-green-600">
                        ({validationData.length} targets across{' '}
                        {new Set(validationData.map(d => d.area_id)).size} areas)
                      </span>
                    </div>
                    <button
                      onClick={() => { setValidationData([]); setValidationFilename(''); }}
                      className="text-xs text-gray-500 hover:text-red-600"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ) : (
                <div
                  className={`px-6 py-8 text-center border-2 border-dashed m-4 rounded-lg transition-colors ${
                    valDragOver ? 'border-blue-400 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onDragOver={(e) => { e.preventDefault(); setValDragOver(true); }}
                  onDragLeave={() => setValDragOver(false)}
                  onDrop={handleValFileDrop}
                >
                  <Upload size={24} className="mx-auto mb-2 text-gray-400" />
                  <p className="text-sm text-gray-600 mb-2">
                    Drop <span className="font-mono text-gray-800">validation_results.csv</span> here
                  </p>
                  <label className="inline-block bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 px-4 py-1.5 rounded text-sm cursor-pointer">
                    Choose file
                    <input
                      type="file"
                      accept=".csv"
                      className="hidden"
                      onChange={handleValFileSelect}
                    />
                  </label>
                </div>
              )}
            </div>

            {/* View Dashboard button */}
            {(hasCalibration || hasValidation) && (
              <button
                onClick={() => setShowDashboard(true)}
                className="w-full bg-green-600 hover:bg-green-700 text-white py-3 rounded-lg text-lg font-semibold shadow-lg transition-colors"
              >
                View dashboard
                {hasBoth && ' (calibration + validation)'}
                {hasCalibration && !hasValidation && ' (calibration only)'}
                {!hasCalibration && hasValidation && ' (validation only)'}
              </button>
            )}
          </div>
        )}

        {/* Dashboard Content */}
        {showDashboard && (
          <div className="space-y-6">
            {/* Top Bar: Tabs + Actions */}
            <div className="flex items-center justify-between">
              {/* View Mode Tabs */}
              {availableTabs.length > 1 && (
                <div className="flex border border-gray-300 rounded-md overflow-hidden">
                  {availableTabs.map(tab => (
                    <button
                      key={tab}
                      onClick={() => setViewMode(tab)}
                      className={`px-4 py-2 text-sm font-medium ${
                        viewMode === tab
                          ? 'bg-blue-600 text-white'
                          : 'bg-white text-gray-700 hover:bg-gray-50'
                      }`}
                    >
                      {tab === 'calibration' ? 'Calibration' : tab === 'validation' ? 'Validation' : 'Explorer'}
                    </button>
                  ))}
                </div>
              )}
              {availableTabs.length <= 1 && <div />}

              {/* Action Buttons */}
              <div className="flex gap-3">
                {(githubArtifactInfo || deeplinkParams) && (
                  <button
                    onClick={handleShare}
                    className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md text-sm font-medium flex items-center gap-2"
                  >
                    <Share size={16} />
                    Share dashboard
                  </button>
                )}
                {!hasValidation && hasCalibration && (
                  <label className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-md text-sm font-medium cursor-pointer flex items-center gap-2">
                    <Upload size={14} />
                    Add validation
                    <input
                      type="file"
                      accept=".csv"
                      className="hidden"
                      onChange={handleValFileSelect}
                    />
                  </label>
                )}
                <button
                  onClick={resetAll}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium"
                >
                  Start over
                </button>
              </div>
            </div>

            {/* Calibration Tab */}
            {viewMode === 'calibration' && hasCalibration && (
              <>
                {comparisonMode ? (
                  <>
                    <ComparisonQualitySummary
                      firstData={data}
                      secondData={secondData}
                      firstName={filename}
                      secondName={secondFilename}
                    />
                    <ComparisonSummary
                      firstData={data}
                      secondData={secondData}
                      firstName={filename}
                      secondName={secondFilename}
                    />
                    <ComparisonCharts
                      firstData={data}
                      secondData={secondData}
                      firstName={filename}
                      secondName={secondFilename}
                    />
                    <RegressionAnalysis
                      firstData={data}
                      secondData={secondData}
                      firstName={filename}
                      secondName={secondFilename}
                    />
                    <TargetConvergenceComparison
                      firstData={data}
                      secondData={secondData}
                      firstName={filename}
                      secondName={secondFilename}
                    />
                    <ComparisonDataTable
                      firstData={data}
                      secondData={secondData}
                      firstName={filename}
                      secondName={secondFilename}
                    />
                  </>
                ) : (
                  <>
                    <MetricsOverview data={data} />
                    <ErrorDistribution data={data} />
                    <CalibrationSummary data={data} />
                    <LossChart data={data} />
                    <SingleDatasetBarChart data={data} />
                    <DataTable data={data} />
                  </>
                )}
              </>
            )}

            {/* Validation Tab */}
            {viewMode === 'validation' && hasValidation && (
              <>
                <SanityCheckPanel data={validationData} />
                <ValidationSummary data={validationData} />
                <ValidationScatterPlot data={validationData} />
                <ValidationByAreaChart data={validationData} />
                <ValidationDataTable data={validationData} />
              </>
            )}

            {/* Explorer Tab */}
            {viewMode === 'explorer' && hasBoth && (
              <>
                <SanityCheckPanel data={validationData} />
                <CalibrationVsSimComparison
                  calibrationData={data}
                  validationData={validationData}
                />
                <StateRollupComparison
                  calibrationData={data}
                  validationData={validationData}
                />
                <UnifiedResultsTable
                  calibrationData={data}
                  validationData={validationData}
                />
              </>
            )}

            {/* Fallback if no data for current tab */}
            {((viewMode === 'calibration' && !hasCalibration) ||
              (viewMode === 'validation' && !hasValidation)) && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
                <div className="text-yellow-800">
                  <h3 className="text-sm font-medium">No data loaded</h3>
                  <p className="text-sm mt-1">
                    Load a {viewMode === 'calibration' ? 'calibration_log.csv' : 'validation_results.csv'} file to see this tab.
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
