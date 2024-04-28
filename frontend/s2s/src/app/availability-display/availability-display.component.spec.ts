import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AvailabilityDisplayComponent } from './availability-display.component';

describe('AvailabilityDisplayComponent', () => {
  let component: AvailabilityDisplayComponent;
  let fixture: ComponentFixture<AvailabilityDisplayComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [AvailabilityDisplayComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(AvailabilityDisplayComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});