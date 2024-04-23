import { Component } from '@angular/core';
import { FormBuilder, FormGroup, FormControl } from '@angular/forms';

// services
import { UserService } from '../_services/user.service';

// helpers
import { availableTimes, days, timeCategoryMap } from '../_helpers/preferences';

/**
 * EventAvailabilitySurveyComponent
 * 
 * This component handles the survey for users to input their availability for 
 * events. It allows users to select their availability for different time 
 * categories on weekdays and weekends.
 * 
 * Example:
 * ```
 * <app-event-availability-survey></app-event-availability-survey>
 * ```
 * 
 * @see UserService
 */
@Component({
  selector: 'app-event-availability-survey',
  templateUrl: './event-availability-survey.component.html',
  styleUrl: './event-availability-survey.component.css'
})
export class EventAvailabilitySurveyComponent {
  availableTimes = availableTimes;
  timeCategoryMap = timeCategoryMap;
  days = days;

  // general availability form
  generalAvailabilityForm: FormGroup = this.fb.group({
    weekdayEarlyMorning: new FormControl(false),
    weekdayMorning: new FormControl(false),
    weekdayAfternoon: new FormControl(false),
    weekdayEvening: new FormControl(false),
    weekdayNight: new FormControl(false),
    weekdayLateNight: new FormControl(false),
    weekdayUnavailable: new FormControl(false),
    weekendEarlyMorning: new FormControl(false),
    weekendMorning: new FormControl(false),
    weekendAfternoon: new FormControl(false),
    weekendEvening: new FormControl(false),
    weekendNight: new FormControl(false),
    weekendLateNight: new FormControl(false),
    weekendUnavailable: new FormControl(false),
  });

  // general availability controls for weekdays
  weekdayGeneralAvailabilityControls = [
    'weekdayEarlyMorning',
    'weekdayMorning',
    'weekdayAfternoon',
    'weekdayEvening',
    'weekdayNight',
    'weekdayLateNight',
    'weekdayUnavailable'
  ];

  // general availability controls for weekends
  weekendGeneralAvailabilityControls = [
    'weekendEarlyMorning',
    'weekendMorning',
    'weekendAfternoon',
    'weekendEvening',
    'weekendNight',
    'weekendLateNight',
    'weekendUnavailable'
  ];

  // form control label map list for weekdays
  weekdayFormControlLabelMapList = [
    {control: 'weekdayUnavailable', label: 'Unavailable'},
    {control: 'weekdayEarlyMorning', label: 'Early morning (5-8a)'},
    {control: 'weekdayMorning', label: 'Morning (9a-12p)'},
    {control: 'weekdayAfternoon', label: 'Afternoon (1-4p)'},
    {control: 'weekdayEvening', label: 'Evening (5-8p)'},
    {control: 'weekdayNight', label: 'Night (9p-12a)'},
    {control: 'weekdayLateNight', label: 'Late night (1-4a)'}
  ];

  // form control label map list for weekends
  weekendFormControlLabelMapList = [
    {control: 'weekendUnavailable', label: 'Unavailable'},
    {control: 'weekendEarlyMorning', label: 'Early morning (5-8a)'},
    {control: 'weekendMorning', label: 'Morning (9a-12p)'},
    {control: 'weekendAfternoon', label: 'Afternoon (1-4p)'},
    {control: 'weekendEvening', label: 'Evening (5-8p)'},
    {control: 'weekendNight', label: 'Night (9p-12a)'},
    {control: 'weekendLateNight', label: 'Late night (1-4a)'}
  ];

  constructor(
    public userService: UserService,
    private fb: FormBuilder
  ) {}

  /**
   * Get the disabled state for the weekday controls.
   * 
   * @param control The control to get the disabled state of.
   * @returns If the control should be disabled.
   */
  getWeekdayDisabledState(control: string) {
    // determine the state of the unavailable control
    if (control.toLowerCase().includes('unavailable')) {
      for (let weekdayControl of this.weekdayGeneralAvailabilityControls) {
        if (this.generalAvailabilityForm.get(weekdayControl)?.value && weekdayControl !== control) {
          return true;
        }
      }
    } 
    
    // determine the state of all other controls
    else {
      return this.generalAvailabilityForm.get('weekdayUnavailable')?.value
    }
  }

  /**
   * Get the disabled state for the weekend controls.
   * 
   * @param control The control to get the disabled state of.
   * @returns If the control should be disabled.
   */
  getWeekendDisabledState(control: string) {
    // determine the state of the unavailable control
    if (control.toLowerCase().includes('unavailable')) {
      for (let weekendControl of this.weekendGeneralAvailabilityControls) {
        if (this.generalAvailabilityForm.get(weekendControl)?.value && weekendControl !== control) {
          return true;
        }
      }
    } 
    
    // determine the state of all other controls
    else {
      return this.generalAvailabilityForm.get('weekendUnavailable')?.value
    }
  }

  /**
   * Adds the time range to the availability.
   * 
   * @param timeRange the time range to add
   * @param days the days to add the time range to
   */
  addTimeRange(timeRange: string, days: string[]) {
    let times = this.timeCategoryMap[timeRange];
      for (let day of days) {
        // get the current availability for the day
        let currentTimes = this.userService.eventAvailabilityForm.value[day];

        // remove the unavailable time
        currentTimes = currentTimes.filter((time: number) => time !== 0);

        // add these times to the current availability
        times.forEach((time: number) => {
          if (!currentTimes.includes(time)) {
            currentTimes.push(time);
          }
        });

        // update the availability for the day
        this.userService.eventAvailabilityForm.controls[day].setValue(currentTimes);
      } 
  }

  /**
   * Removes the time range from the availability.
   * 
   * @param timeRange the time range to remove
   * @param days the days to remove the time range from
   * @returns the updated availability
   */
  removeTimeRange(timeRange: string, days: string[]) {
    let times = this.timeCategoryMap[timeRange];
    let currentTimes: number[] = [];
    for (let day of days) {
      // get the current availability for the day
      let currentTimes = this.userService.eventAvailabilityForm.value[day];

      // remove the times from the current availability
      times.forEach((time: number) => {
        currentTimes = currentTimes.filter((t: number) => t !== time);
      });

      // update the form control
      this.userService.eventAvailabilityForm.controls[day].setValue(currentTimes);
    }
    return currentTimes;
  }


  /**
   * Updates the times for the weekdays.
   */
  updateWeekdayTimes() {
    let weekdayTimes: string[] = ['mondayTimes', 'tuesdayTimes', 'wednesdayTimes', 'thursdayTimes', 'fridayTimes'];
    for (let controlLabel of this.weekdayFormControlLabelMapList) {
      // available
      if (this.generalAvailabilityForm.value[controlLabel.control]) {
        this.addTimeRange(controlLabel.label, weekdayTimes);
      } 
      
      // unavailable
      else {
        this.removeTimeRange(controlLabel.label, weekdayTimes);
      }
    }
  }

  /**
   * Updates the times for the weekends.
   */
  updateWeekendTimes() {
    let weekendTimes: string[] = ['saturdayTimes', 'sundayTimes'];
    for (let controlLabel of this.weekendFormControlLabelMapList) {
      // available
      if (this.generalAvailabilityForm.value[controlLabel.control]) {
        this.addTimeRange(controlLabel.label, weekendTimes);
      } 
      
      // unavailable
      else {
        this.removeTimeRange(controlLabel.label, weekendTimes);
      }
    }
  }

  /**
   * Submits the event availability form.
   */
  onSubmit() {
    console.log(this.userService.eventAvailabilityForm.value);
  }
}
