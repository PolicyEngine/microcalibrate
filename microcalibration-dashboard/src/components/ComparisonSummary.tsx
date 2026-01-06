'use client';

import { CalibrationDataPoint } from '@/types/calibration';
import { BarChart3, Target } from 'lucide-react';

interface ComparisonSummaryProps {
  firstData: CalibrationDataPoint[];
  secondData: CalibrationDataPoint[];
  firstName: string;
  secondName: string;
}

interface DatasetSummary {
  totalTargets: number;
  uniqueTargets: Set<string>;
  excludedTargets: string[];
  epochs: number[];
  maxEpoch: number;
  minEpoch: number;
  excellent: number; // < 5% error
  good: number; // 5-20% error
  needWork: number; // > 20% error
  avgFinalError: number; // average relative error in final epoch
  progressSummary: {
    improved: number;
    worsened: number;
    stable: number;
  };
}

export default function ComparisonSummary({ firstData, secondData, firstName, secondName }: ComparisonSummaryProps) {
  // Analyze both datasets
  const analyzeDataset = (data: CalibrationDataPoint[]): DatasetSummary => {
    const uniqueTargets = new Set(data.map(d => d.target_name));
    const epochs = Array.from(new Set(data.map(d => d.epoch))).sort((a, b) => a - b);
    
    // Identify excluded targets (those with constant estimates across epochs)
    const excludedTargets: string[] = [];
    const targetGroups = new Map<string, CalibrationDataPoint[]>();
    
    // Group by target name
    data.forEach(point => {
      if (!targetGroups.has(point.target_name)) {
        targetGroups.set(point.target_name, []);
      }
      targetGroups.get(point.target_name)!.push(point);
    });
    
    targetGroups.forEach((points, targetName) => {
      if (points.length > 1) {
        const estimates = points.map(p => p.estimate);
        const isConstant = estimates.every(est => Math.abs(est - estimates[0]) < 1e-6);
        if (isConstant) {
          excludedTargets.push(targetName);
        }
      }
    });
    
    // Get final epoch data for quality assessment
    const maxEpoch = Math.max(...epochs);
    const minEpoch = Math.min(...epochs);
    const finalEpochData = data.filter(d => d.epoch === maxEpoch);
    const initialEpochData = data.filter(d => d.epoch === minEpoch);
    
    let excellent = 0, good = 0, needWork = 0;
    let totalFinalError = 0, validFinalCount = 0;
    
    finalEpochData.forEach(d => {
      if (d.rel_abs_error !== undefined && !isNaN(d.rel_abs_error)) {
        if (d.rel_abs_error < 0.05) excellent++;
        else if (d.rel_abs_error < 0.20) good++;
        else needWork++;
        
        totalFinalError += d.rel_abs_error;
        validFinalCount++;
      }
    });
    
    // Calculate progress summary (initial vs final epoch)
    let improved = 0, worsened = 0, stable = 0;
    
    uniqueTargets.forEach(targetName => {
      const initialTarget = initialEpochData.find(d => d.target_name === targetName);
      const finalTarget = finalEpochData.find(d => d.target_name === targetName);
      
      if (initialTarget && finalTarget && 
          initialTarget.rel_abs_error !== undefined && finalTarget.rel_abs_error !== undefined &&
          !isNaN(initialTarget.rel_abs_error) && !isNaN(finalTarget.rel_abs_error) &&
          !excludedTargets.includes(targetName)) { // Exclude excluded targets from progress calculations
        
        const improvement = initialTarget.rel_abs_error - finalTarget.rel_abs_error;
        const relativeImprovement = initialTarget.rel_abs_error > 0 ? improvement / initialTarget.rel_abs_error : 0;
        
        if (relativeImprovement > 0.2) improved++;
        else if (relativeImprovement < -0.1) worsened++;
        else stable++;
      }
    });
    
    return {
      totalTargets: uniqueTargets.size,
      uniqueTargets,
      excludedTargets,
      epochs,
      maxEpoch,
      minEpoch,
      excellent,
      good,
      needWork,
      avgFinalError: validFinalCount > 0 ? totalFinalError / validFinalCount : 0,
      progressSummary: { improved, worsened, stable }
    };
  };

  const firstSummary = analyzeDataset(firstData);
  const secondSummary = analyzeDataset(secondData);
  
  // Find overlapping and unique targets
  const overlappingTargets = new Set([...firstSummary.uniqueTargets].filter(x => secondSummary.uniqueTargets.has(x)));
  const firstOnlyTargets = new Set([...firstSummary.uniqueTargets].filter(x => !secondSummary.uniqueTargets.has(x)));
  const secondOnlyTargets = new Set([...secondSummary.uniqueTargets].filter(x => !firstSummary.uniqueTargets.has(x)));
  
  // Find excluded targets overlap
  const firstExcluded = new Set(firstSummary.excludedTargets);
  const secondExcluded = new Set(secondSummary.excludedTargets);
  const overlappingExcluded = new Set([...firstExcluded].filter(x => secondExcluded.has(x)));
  const firstOnlyExcluded = new Set([...firstExcluded].filter(x => !secondExcluded.has(x)));
  const secondOnlyExcluded = new Set([...secondExcluded].filter(x => !firstExcluded.has(x)));

  return (
    <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
      <h2 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
        <BarChart3 className="w-5 h-5 mr-2" />
        Dataset comparison summary
      </h2>

      {/* Overview stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* First dataset */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-blue-800 mb-3">First dataset</h3>
          <p className="text-sm text-blue-700 mb-3 truncate" title={firstName}>{firstName}</p>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-blue-600">Targets:</span>
              <span className="font-medium text-blue-800">{firstSummary.totalTargets}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-blue-600">Epochs:</span>
              <span className="font-medium text-blue-800">{firstSummary.minEpoch} - {firstSummary.maxEpoch} ({firstSummary.epochs.length} logged)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-blue-600">Avg final error:</span>
              <span className="font-medium text-blue-800">{(firstSummary.avgFinalError * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-blue-600">Excluded targets:</span>
              <span className="font-medium text-blue-800">
                {firstSummary.excludedTargets.length > 0 ? firstSummary.excludedTargets.length : 'None'}
              </span>
            </div>
          </div>

          {/* Quality distribution */}
          <div className="mt-4 pt-3 border-t border-blue-200">
            <p className="text-xs font-medium text-blue-600 mb-2">Final epoch quality:</p>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-green-600">Excellent (&lt;5%):</span>
                <span className="font-medium">{firstSummary.excellent}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-yellow-600">Good (5-20%):</span>
                <span className="font-medium">{firstSummary.good}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-red-600">Need work (&gt;20%):</span>
                <span className="font-medium">{firstSummary.needWork}</span>
              </div>
            </div>
          </div>

          {/* Progress Summary */}
          <div className="mt-4 pt-3 border-t border-blue-300">
            <p className="text-xs font-medium text-blue-600 mb-2">Calibration progress (initial → final):</p>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-green-600">Significantly improved (&gt;20%):</span>
                <span className="font-medium">{firstSummary.progressSummary.improved}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-red-600">Worsened (&gt;10%):</span>
                <span className="font-medium">{firstSummary.progressSummary.worsened}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Minimal change:</span>
                <span className="font-medium">{firstSummary.progressSummary.stable}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Second dataset */}
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-purple-800 mb-3">Second dataset</h3>
          <p className="text-sm text-purple-700 mb-3 truncate" title={secondName}>{secondName}</p>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-purple-600">Targets:</span>
              <span className="font-medium text-purple-800">{secondSummary.totalTargets}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-purple-600">Epochs:</span>
              <span className="font-medium text-purple-800">{secondSummary.minEpoch} - {secondSummary.maxEpoch} ({secondSummary.epochs.length} logged)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-purple-600">Avg final error:</span>
              <span className="font-medium text-purple-800">{(secondSummary.avgFinalError * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-purple-600">Excluded targets:</span>
              <span className="font-medium text-purple-800">
                {secondSummary.excludedTargets.length > 0 ? secondSummary.excludedTargets.length : 'None'}
              </span>
            </div>
          </div>

          {/* Quality distribution */}
          <div className="mt-4 pt-3 border-t border-purple-200">
            <p className="text-xs font-medium text-purple-600 mb-2">Final epoch quality:</p>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-green-600">Excellent (&lt;5%):</span>
                <span className="font-medium">{secondSummary.excellent}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-yellow-600">Good (5-20%):</span>
                <span className="font-medium">{secondSummary.good}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-red-600">Need work (&gt;20%):</span>
                <span className="font-medium">{secondSummary.needWork}</span>
              </div>
            </div>
          </div>

          {/* Progress Summary */}
          <div className="mt-4 pt-3 border-t border-purple-300">
            <p className="text-xs font-medium text-purple-600 mb-2">Calibration progress (initial → final):</p>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-green-600">Significantly improved (&gt;20%):</span>
                <span className="font-medium">{secondSummary.progressSummary.improved}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-red-600">Worsened (&gt;10%):</span>
                <span className="font-medium">{secondSummary.progressSummary.worsened}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Minimal change:</span>
                <span className="font-medium">{secondSummary.progressSummary.stable}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Target overlap analysis */}
      <div className="border-t border-gray-200 pt-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
          <Target className="w-4 h-4 mr-2" />
          Target overlap analysis
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Overlapping targets */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-center mb-2">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
              <h4 className="font-semibold text-green-800">Common targets</h4>
            </div>
            <div className="text-2xl font-bold text-green-700">{overlappingTargets.size}</div>
            <div className="text-xs text-green-600">
              {((overlappingTargets.size / Math.max(firstSummary.totalTargets, secondSummary.totalTargets)) * 100).toFixed(1)}% of max dataset
            </div>
          </div>

          {/* First dataset only */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center mb-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
              <h4 className="font-semibold text-blue-800">First only</h4>
            </div>
            <div className="text-2xl font-bold text-blue-700">{firstOnlyTargets.size}</div>
            <div className="text-xs text-blue-600">
              Targets unique to first dataset
            </div>
          </div>

          {/* Second dataset only */}
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <div className="flex items-center mb-2">
              <div className="w-3 h-3 bg-purple-500 rounded-full mr-2"></div>
              <h4 className="font-semibold text-purple-800">Second only</h4>
            </div>
            <div className="text-2xl font-bold text-purple-700">{secondOnlyTargets.size}</div>
            <div className="text-xs text-purple-600">
              Targets unique to second dataset
            </div>
          </div>
        </div>

        {/* Summary */}
        <div className="mt-4 p-3 bg-gray-50 border border-gray-200 rounded">
          <p className="text-sm text-gray-700">
            <strong>Analysis:</strong> {overlappingTargets.size} targets can be directly compared between datasets.
            {firstOnlyTargets.size > 0 && ` ${firstOnlyTargets.size} targets were removed in the second run.`}
            {secondOnlyTargets.size > 0 && ` ${secondOnlyTargets.size} targets were added in the second run.`}
          </p>
        </div>
      </div>

      {/* Excluded targets analysis */}
      {(firstExcluded.size > 0 || secondExcluded.size > 0) && (
        <div className="border-t border-gray-200 pt-6 mt-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Excluded targets analysis
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Common excluded */}
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
              <div className="flex items-center mb-2">
                <div className="w-3 h-3 bg-orange-500 rounded-full mr-2"></div>
                <h4 className="font-semibold text-orange-800">Both excluded</h4>
              </div>
              <div className="text-2xl font-bold text-orange-700">{overlappingExcluded.size}</div>
              <div className="text-xs text-orange-600">
                {overlappingExcluded.size <= 3 
                  ? [...overlappingExcluded].join(', ') || 'None'
                  : `${[...overlappingExcluded].slice(0, 3).join(', ')}, +${overlappingExcluded.size - 3} more`}
              </div>
            </div>

            {/* First dataset excluded only */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center mb-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
                <h4 className="font-semibold text-blue-800">First excluded only</h4>
              </div>
              <div className="text-2xl font-bold text-blue-700">{firstOnlyExcluded.size}</div>
              <div className="text-xs text-blue-600">
                {firstOnlyExcluded.size <= 3 
                  ? [...firstOnlyExcluded].join(', ') || 'None'
                  : `${[...firstOnlyExcluded].slice(0, 3).join(', ')}, +${firstOnlyExcluded.size - 3} more`}
              </div>
            </div>

            {/* Second dataset excluded only */}
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <div className="flex items-center mb-2">
                <div className="w-3 h-3 bg-purple-500 rounded-full mr-2"></div>
                <h4 className="font-semibold text-purple-800">Second excluded only</h4>
              </div>
              <div className="text-2xl font-bold text-purple-700">{secondOnlyExcluded.size}</div>
              <div className="text-xs text-purple-600">
                {secondOnlyExcluded.size <= 3 
                  ? [...secondOnlyExcluded].join(', ') || 'None'
                  : `${[...secondOnlyExcluded].slice(0, 3).join(', ')}, +${secondOnlyExcluded.size - 3} more`}
              </div>
            </div>
          </div>

          {/* Excluded targets summary */}
          <div className="mt-4 p-3 bg-orange-50 border border-orange-200 rounded">
            <p className="text-sm text-orange-700">
              <strong>Excluded targets note:</strong> These targets were held constant during calibration and appear in logs with their initial estimates for reference.
              {overlappingExcluded.size > 0 && ` ${overlappingExcluded.size} targets were excluded in both runs.`}
              {firstOnlyExcluded.size > 0 && ` ${firstOnlyExcluded.size} targets were excluded only in the first run.`}
              {secondOnlyExcluded.size > 0 && ` ${secondOnlyExcluded.size} targets were excluded only in the second run.`}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
