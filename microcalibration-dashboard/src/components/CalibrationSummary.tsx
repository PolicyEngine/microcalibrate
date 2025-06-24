'use client';

import { CalibrationDataPoint } from '@/types/calibration';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface CalibrationSummaryProps {
  data: CalibrationDataPoint[];
}

interface TargetSummary {
  target_name: string;
  initialError: number;
  finalError: number;
  improvement: number;
  category: 'improved_significantly' | 'worsened' | 'minimal_change';
}

export default function CalibrationSummary({ data }: CalibrationSummaryProps) {
  if (data.length === 0) {
    return null;
  }

  // Group data by target and calculate improvement metrics
  const targetSummaries: TargetSummary[] = [];
  const targetGroups = new Map<string, CalibrationDataPoint[]>();
  
  // Group by target name
  data.forEach(point => {
    if (!targetGroups.has(point.target_name)) {
      targetGroups.set(point.target_name, []);
    }
    targetGroups.get(point.target_name)!.push(point);
  });

  // Calculate metrics for each target
  targetGroups.forEach((points, targetName) => {
    // Sort by epoch to get initial and final states
    const sortedPoints = points.sort((a, b) => a.epoch - b.epoch);
    const initialPoint = sortedPoints[0];
    const finalPoint = sortedPoints[sortedPoints.length - 1];
    
    if (initialPoint && finalPoint && 
        initialPoint.rel_abs_error !== undefined && finalPoint.rel_abs_error !== undefined &&
        !isNaN(initialPoint.rel_abs_error) && !isNaN(finalPoint.rel_abs_error)) {
      
      const initialError = initialPoint.rel_abs_error;
      const finalError = finalPoint.rel_abs_error;
      const improvement = initialError - finalError; // Positive = improvement, negative = worsened
      const relativeImprovement = initialError > 0 ? improvement / initialError : 0;
      
      let category: 'improved_significantly' | 'worsened' | 'minimal_change';
      
      if (relativeImprovement > 0.2) { // Improved by more than 20%
        category = 'improved_significantly';
      } else if (relativeImprovement < -0.1) { // Worsened by more than 10%
        category = 'worsened';
      } else {
        category = 'minimal_change';
      }
      
      targetSummaries.push({
        target_name: targetName,
        initialError,
        finalError,
        improvement,
        category
      });
    }
  });

  // Calculate statistics
  const totalTargets = targetSummaries.length;
  const improvedSignificantly = targetSummaries.filter(t => t.category === 'improved_significantly').length;
  const worsened = targetSummaries.filter(t => t.category === 'worsened').length;
  const minimalChange = targetSummaries.filter(t => t.category === 'minimal_change').length;

  const improvedPercentage = totalTargets > 0 ? (improvedSignificantly / totalTargets * 100) : 0;
  const worsenedPercentage = totalTargets > 0 ? (worsened / totalTargets * 100) : 0;
  const minimalPercentage = totalTargets > 0 ? (minimalChange / totalTargets * 100) : 0;

  // Get examples for each category
  const improvedExamples = targetSummaries
    .filter(t => t.category === 'improved_significantly')
    .sort((a, b) => b.improvement - a.improvement)
    .slice(0, 3);
    
  const worsenedExamples = targetSummaries
    .filter(t => t.category === 'worsened')
    .sort((a, b) => a.improvement - b.improvement)
    .slice(0, 3);
    
  const minimalExamples = targetSummaries
    .filter(t => t.category === 'minimal_change')
    .slice(0, 3);

  return (
    <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
      <h2 className="text-xl font-bold text-gray-800 mb-4">Calibration Summary</h2>
      <p className="text-gray-600 mb-6">
        Analysis of how calibration affected each target&apos;s accuracy from initial to final epoch
      </p>

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {/* Improved Significantly */}
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center mb-2">
            <TrendingUp className="w-5 h-5 text-green-600 mr-2" />
            <h3 className="text-lg font-semibold text-green-800">Significantly Improved</h3>
          </div>
          <div className="text-2xl font-bold text-green-700">{improvedSignificantly}</div>
          <div className="text-sm text-green-600">
            {improvedPercentage.toFixed(1)}% of targets ({improvedSignificantly}/{totalTargets})
          </div>
          <div className="text-xs text-green-500 mt-1">
            Reduced error by &gt;20%
          </div>
        </div>

        {/* Worsened */}
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center mb-2">
            <TrendingDown className="w-5 h-5 text-red-600 mr-2" />
            <h3 className="text-lg font-semibold text-red-800">Worsened</h3>
          </div>
          <div className="text-2xl font-bold text-red-700">{worsened}</div>
          <div className="text-sm text-red-600">
            {worsenedPercentage.toFixed(1)}% of targets ({worsened}/{totalTargets})
          </div>
          <div className="text-xs text-red-500 mt-1">
            Increased error by &gt;10%
          </div>
        </div>

        {/* Minimal Change */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <div className="flex items-center mb-2">
            <Minus className="w-5 h-5 text-gray-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-800">Minimal Change</h3>
          </div>
          <div className="text-2xl font-bold text-gray-700">{minimalChange}</div>
          <div className="text-sm text-gray-600">
            {minimalPercentage.toFixed(1)}% of targets ({minimalChange}/{totalTargets})
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Error change within ±20%/±10%
          </div>
        </div>
      </div>

      {/* Detailed Examples */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Top Improvements */}
        {improvedExamples.length > 0 && (
          <div>
            <h4 className="text-md font-semibold text-green-800 mb-3 flex items-center">
              <TrendingUp className="w-4 h-4 mr-1" />
              Top Improvements
            </h4>
            <div className="space-y-2">
              {improvedExamples.map((target, i) => (
                <div key={i} className="bg-green-50 p-3 rounded border">
                  <div className="text-sm font-medium text-gray-800 truncate" title={target.target_name}>
                    {target.target_name}
                  </div>
                  <div className="text-xs text-green-700">
                    {(target.initialError * 100).toFixed(1)}% → {(target.finalError * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-green-600">
                    -{(target.improvement * 100).toFixed(1)}pp improvement
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Worsened Cases */}
        {worsenedExamples.length > 0 && (
          <div>
            <h4 className="text-md font-semibold text-red-800 mb-3 flex items-center">
              <TrendingDown className="w-4 h-4 mr-1" />
              Worsened Cases
            </h4>
            <div className="space-y-2">
              {worsenedExamples.map((target, i) => (
                <div key={i} className="bg-red-50 p-3 rounded border">
                  <div className="text-sm font-medium text-gray-800 truncate" title={target.target_name}>
                    {target.target_name}
                  </div>
                  <div className="text-xs text-red-700">
                    {(target.initialError * 100).toFixed(1)}% → {(target.finalError * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-red-600">
                    +{Math.abs(target.improvement * 100).toFixed(1)}pp worsened
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Stable Cases */}
        {minimalExamples.length > 0 && (
          <div>
            <h4 className="text-md font-semibold text-gray-800 mb-3 flex items-center">
              <Minus className="w-4 h-4 mr-1" />
              Stable Cases
            </h4>
            <div className="space-y-2">
              {minimalExamples.map((target, i) => (
                <div key={i} className="bg-gray-50 p-3 rounded border">
                  <div className="text-sm font-medium text-gray-800 truncate" title={target.target_name}>
                    {target.target_name}
                  </div>
                  <div className="text-xs text-gray-700">
                    {(target.initialError * 100).toFixed(1)}% → {(target.finalError * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-600">
                    {target.improvement >= 0 ? '-' : '+'}{Math.abs(target.improvement * 100).toFixed(1)}pp change
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}